import json
from json import decoder
import logging
import os
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoTokenizer, DPRQuestionEncoder,
                          GPT2LMHeadModel)
import random
import numpy as np
from .data import pad_ids
import torch.distributed as dist
logger = logging.getLogger(__name__)
EPSILON = 1e-10


def ReplaceNAN(x, val=0):
    x[x != x] = val
    return x

def MergeEmbed(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])
    total_embeddding = []
    for i in range(batch_size):
        all_docs_embeds = []
        s = set()
        for j in range(topk):
            id1, id2 = prior_topk_documents_ids[i][j], posterior_topk_documents_ids[i][j]
            if id1 not in s:
                s.add(id1)
                all_docs_embeds.append(prior_topk_documents_embeddings[i][j])
            if id2 not in s:
                s.add(id2)
                all_docs_embeds.append(posterior_topk_documents_embeddings[i][j])

        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda() # (H,N)
    return KL

def GetUnionKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = []
        s = set()
        for j in range(topk):
            id1, id2 = prior_topk_documents_ids[i][j], posterior_topk_documents_ids[i][j]
            if id1 not in s:
                s.add(id1)
                all_docs_embeds.append(prior_topk_documents_embeddings[i][j])
            if id2 not in s:
                s.add(id2)
                all_docs_embeds.append(posterior_topk_documents_embeddings[i][j])


        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda() # (H,N)

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds # (1, H)*(H,N): (1, N)
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze() # (N)
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


def GetPostKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = posterior_topk_documents_embeddings[i]
        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda()

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze()
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


class MagicalModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(MagicalModel, self).__init__()
        self.max_length = 512

        self.config = AutoConfig.from_pretrained(
            args.question_encoder_model_name)
        self.encoder = DPRQuestionEncoder.from_pretrained(
            args.question_encoder_model_name, config=self.config)
        if dist.get_rank()==0:
            logger.info("Loading magical model from %s", args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def forward(self, batch, topk):
        input_ids, token_type_ids = batch

        # question_embeddings: batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length], token_type_ids=token_type_ids[:, :self.max_length]).pooler_output
        retrieved_indices = self.indexed_passages.retrieve(
            question_embeddings, topk)

        return torch.tensor(retrieved_indices).cuda()


class PriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "prior", args.checkpoint), config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading prior model from %s", os.path.join(args.model_path, "prior", args.checkpoint))
        elif (args.prior_path != None):
            self.config = AutoConfig.from_pretrained(args.prior_path)
            self.tokenizer = AutoTokenizer.from_pretrained(args.prior_path)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.prior_path, config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading prior model from %s", args.prior_path)
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading prior model from %s",
                        args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def forward(self, batch, topk, magic=0):
        # input_ids: batch_size x sequence_length
        if (magic):
            input_ids, retrieved_indices = batch
            retrieved_indices = retrieved_indices.cpu().tolist()
        else:
            input_ids = batch

        # question_embeddings: batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length]).pooler_output

        if (not magic):
            retrieved_indices = self.indexed_passages.retrieve(
                question_embeddings, topk)

        # topk_documents_embeddings: batch_size x topk x 768
        topk_documents_embeddings = self.indexed_passages.get_field_by_indices(
            retrieved_indices, "embeddings")
        topk_documents_embeddings = torch.tensor(
            topk_documents_embeddings).cuda()

        # logits: batch_size x topk
        logits = torch.bmm(topk_documents_embeddings,
                           question_embeddings.unsqueeze(2)).squeeze(2)

        return logits, torch.tensor(retrieved_indices).cuda(), question_embeddings
    
        

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "prior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving prior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class PosteriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PosteriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "posterior", args.checkpoint), config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading posterior model from %s", os.path.join(args.model_path, "posterior", args.checkpoint))
        elif (args.posterior_path != None):
            self.config = AutoConfig.from_pretrained(args.posterior_path)
            self.tokenizer = AutoTokenizer.from_pretrained(args.posterior_path)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.posterior_path, config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading posterior model from %s", args.posterior_path)
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading posterior model from %s", args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def forward(self, batch, topk, magic=0):
        if (magic):
            input_ids, token_type_ids, retrieved_indices = batch
            retrieved_indices = retrieved_indices.cpu().tolist()
        else:
            input_ids, token_type_ids = batch

        # question_embeddings: batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length], token_type_ids=token_type_ids[:, :self.max_length]).pooler_output
        if (not magic):
            retrieved_indices = self.indexed_passages.retrieve(
                question_embeddings, topk)

        # topk_documents_embeddings: batch_size x topk x 768
        topk_documents_embeddings = self.indexed_passages.get_field_by_indices(
            retrieved_indices, "embeddings")
        topk_documents_embeddings = torch.tensor(
            topk_documents_embeddings).cuda()

        # logits: topk x 768
        logits = torch.bmm(topk_documents_embeddings,
                           question_embeddings.unsqueeze(2)).squeeze(2) # batch_size x topk

        return logits, torch.tensor(retrieved_indices).cuda(), question_embeddings

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "posterior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving posterior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class DecoderModel(nn.Module):
    def __init__(self, args):
        super(DecoderModel, self).__init__()

        self.SPECIAL_TOKENS = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<speaker1>", "<speaker2>"]
        }
        self.SPECIAL_TOKENS_VALUES = [
            "<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>"]
        self.max_length = 512
        self.divide = args.divide
        self.multitask = args.multitask

        if (self.multitask):
            self.SPECIAL_TOKENS["cls_token"] = "[CLS]"
            self.SPECIAL_TOKENS_VALUES.append("[CLS]")

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(
                args.model_path, "decoder", args.checkpoint), config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading decoder model from %s", os.path.join(args.model_path, "decoder", args.checkpoint))
        elif (args.decoder_path != None):
            self.config = AutoConfig.from_pretrained(args.decoder_path)
            self.tokenizer = AutoTokenizer.from_pretrained(args.decoder_path)
            self.decoder = GPT2LMHeadModel.from_pretrained(
                args.decoder_path, config=self.config)
            if dist.get_rank()==0:
                logger.info("Loading decoder model from %s", args.decoder_path)
        else:
            self.config = AutoConfig.from_pretrained(args.decoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.decoder_model_name)
            self.decoder = GPT2LMHeadModel.from_pretrained(
                args.decoder_model_name, config=self.config)

            self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
            self.decoder.resize_token_embeddings(len(self.tokenizer))
            if dist.get_rank()==0:
                logger.info("Loading decoder model from %s", args.decoder_model_name)

        self.bos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2 = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"])

        if (self.multitask):
            self.cls = self.tokenizer.convert_tokens_to_ids(
                self.SPECIAL_TOKENS["cls_token"])

            self.classifier = nn.Linear(768, 1)

    def _dec(self, text):
        input_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text))
        return input_ids

    def _prepare_inputs(self,
                        decoder_input_ids,
                        decoder_response_ids,
                        topk_documents_text,
                        with_eos=True):
        # decoder_input_ids: batch_size x sequence_length
        # decoder_response_ids: batch_size x sequence_length
        # topk_documents_text: batch_size x topk x text_length
        batch_size = len(decoder_input_ids)
        topk = len(topk_documents_text[0])

        list_ids = []
        list_lms = []
        list_type_ids = []
        if (self.multitask):
            cls_index = []
        for i in range(batch_size):
            for j in range(topk):
                knowledge = self._dec(topk_documents_text[i][j])
                history = decoder_input_ids[i]
                response = decoder_response_ids[i]

                sequence = [[self.bos] + knowledge] + [history] + \
                    [response + ([self.eos] if with_eos else [])]
                sequence_with_speaker = [[self.speaker1 if (len(
                    sequence) - i) % 2 == 0 else self.speaker2] + s for i, s in enumerate(sequence[1:])]
                sequence = [sequence[0]] + sequence_with_speaker

                type_ids = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(
                    sequence) for _ in s]
                lms = ([-100] * sum(len(s)
                                    for s in sequence[:-1])) + [-100] + sequence[-1][1:]
                ids = list(chain(*sequence))

                if (self.multitask):
                    ids = ids[:self.max_length - 1]
                    lms = lms[:self.max_length - 1]
                    type_ids = type_ids[:self.max_length - 1]

                    ids.append(self.cls)
                    lms.append(self.cls)
                    type_ids.append(self.cls)
                    cls_index.append(len(ids) - 1)

                list_ids.append(ids)
                list_lms.append(lms)
                list_type_ids.append(type_ids)

        list_ids = torch.tensor(pad_ids(list_ids, self.pad))
        list_lms = torch.tensor(pad_ids(list_lms, -100))
        list_type_ids = torch.tensor(pad_ids(list_type_ids, self.pad))

        if (self.multitask):
            cls_index = torch.tensor(cls_index)
            cls_index = cls_index.reshape(batch_size, topk)

        list_ids = list_ids[:, :self.max_length]
        list_lms = list_lms[:, :self.max_length]
        list_type_ids = list_type_ids[:, :self.max_length]

        # decoder_input_ids: batch_size x topk x sequence_length
        # decoder_response_ids: batch_size x topk x sequence_length
        # decoder_token_type_ids: batch_size x topk x sequence_length
        decoder_input_ids = list_ids.reshape(batch_size, topk, -1)
        decoder_response_ids = list_lms.reshape(batch_size, topk, -1)
        decoder_token_type_ids = list_type_ids.reshape(batch_size, topk, -1)

        if (self.multitask):
            return decoder_input_ids, decoder_response_ids, decoder_token_type_ids, cls_index
        return decoder_input_ids, decoder_response_ids, decoder_token_type_ids

    def compute_gen_loss_item(self, lm_logits, labels):
        batch_size, topk, sequence_length, _ = lm_logits.shape

        lm_logits = lm_logits.reshape(batch_size * topk, sequence_length, -1) # (B*K, L, V)
        labels = labels.reshape(batch_size * topk, -1) # (B*K, L)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)) # no reduction, the shape of loss is (B*K*L)

        loss = loss.reshape(batch_size * topk, -1) # (B*K, L)
        shift_labels = shift_labels.reshape(batch_size * topk, -1)

        loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        loss = loss.reshape(batch_size, topk) # (B, K)

        return loss

    def Classify(self, batch):
        batch_size, topk, dimension = batch.shape
        batch = batch.reshape(batch_size * topk, dimension)
        batch = self.classifier(batch)
        batch = batch.reshape(batch_size, topk)
        return batch
    
    def cal_prob(self, input_ids, labels):
        # input_ids: (B, L)
        decoder_model_outputs = self.decoder(input_ids, token_type_ids=None)
        logits = decoder_model_outputs[0] # (B,L,V)
        shift_logits = logits[..., :-1, :].contiguous() # (B, L, V)
        shift_labels = labels[..., 1:].contiguous() # (B, L)
        neg_log_probs = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none').view(shift_labels.size(0), -1) # (B, L)
        neg_log_probs = neg_log_probs.sum(-1) # (B,)
        sent_probs = torch.exp(-neg_log_probs)
        return sent_probs, neg_log_probs

    def forward(self, batch):
        if (self.multitask):
            decoder_input_ids, decoder_response_ids, cls_index = batch
        else:
            decoder_input_ids, decoder_response_ids = batch

        batch_size, topk, sequence_length = decoder_input_ids.shape
        decoder_input_ids = decoder_input_ids.reshape(batch_size * topk, -1) # (B*K, L)

        if (self.multitask):
            decoder_model_outputs = self.decoder(
                input_ids=decoder_input_ids, token_type_ids=None, output_hidden_states=True)
            hidden_states = decoder_model_outputs[2][-1]
            hidden_states = decoder_model_outputs[2][-1].reshape(
                batch_size, topk, sequence_length, 768)

            x = []
            for i in range(cls_index.shape[0]):
                x.append([])
                for j in range(cls_index.shape[1]):
                    x[-1].append(hidden_states[i, j, cls_index[i][j], :])
            x = [torch.stack(i) for i in x]
            x = torch.stack(x)

            classification_logits = self.Classify(x)
        else:
            decoder_model_outputs = self.decoder(
                input_ids=decoder_input_ids, token_type_ids=None)

        lm_logits = decoder_model_outputs[0] # (B*K, L, V)
        lm_logits = lm_logits.reshape(batch_size, topk, sequence_length, -1) # (B, K, L, V)
        lm_logits = lm_logits / self.divide

        loss = self.compute_gen_loss_item(lm_logits, decoder_response_ids) # (B, K)

        if (self.multitask):
            return loss, lm_logits, classification_logits
        return loss, lm_logits

    def generate_from_1_doc(self,
                            args,
                            decoder_input_ids,
                            best_document_decoder_text):
        output_text = None

        if (self.multitask):
            decoder_input_ids_, _, _, cls_index = self._prepare_inputs(
                [decoder_input_ids], [[]], [[best_document_decoder_text]])

            decoder_input_ids_ = decoder_input_ids_.squeeze(1).cuda()
            cls_index = cls_index.cuda()

            decoder_model_outputs = self.decoder(
                input_ids=decoder_input_ids_, token_type_ids=None, output_hidden_states=True)
            hidden_states = decoder_model_outputs[2][-1].unsqueeze(0)

            x = hidden_states[0, 0, cls_index[0][0], :]
            x = x.unsqueeze(0).unsqueeze(0)

            classification_logits = self.Classify(x)
            if (classification_logits > 0):
                output_text = "CANNOTANSWER"

        if (output_text == None):
            decoder_input_ids_, _, _ = self._prepare_inputs(
                [decoder_input_ids], [[]], [[best_document_decoder_text]], with_eos=False)
            decoder_input_ids_ = decoder_input_ids_.squeeze(1).cuda()

            output = self.decoder.generate(
                input_ids=decoder_input_ids_,
                max_length=args.generation_args["max_length"] +
                decoder_input_ids_.shape[1],
                min_length=args.generation_args["min_length"],
                top_k=args.generation_args["top_k"],
                top_p=args.generation_args["top_p"],
                temperature=args.generation_args["temperature"],
                bos_token_id=self.bos,
                eos_token_id=self.eos,
                pad_token_id=self.pad
            )

            output = output[0][decoder_input_ids_.shape[1]:]
            output_text = self.tokenizer.decode(
                output, skip_special_tokens=True)

        return output_text

    def generate_from_k_docs(self,
                             args,
                             decoder_input_ids,
                             topk_documents_decoder_text,
                             prior_dist):
        p_y_given_zx = []
        output_text = None
        p_max = -1
        for i in range(len(prior_dist)):
            text_ = self.generate_from_1_doc(
                args, decoder_input_ids, topk_documents_decoder_text[i])

            decoder_response_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(text_))
            x = self._prepare_inputs([decoder_input_ids], [decoder_response_ids], [
                                     [topk_documents_decoder_text[i]]], with_eos=False)
            if (self.multitask):
                decoder_input_ids_, decoder_response_ids_, _, _ = x
            else:
                decoder_input_ids_, decoder_response_ids_, _ = x

            decoder_loss, _ = self(
                [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

            p_y_given_zx = torch.exp(-decoder_loss).squeeze().cpu().numpy()
            p_y_given_x = p_y_given_zx * prior_dist[i]
            if (p_y_given_x > p_max):
                p_max = p_y_given_x
                output_text = text_

        if (output_text == None):
            output_text = ""
        return output_text

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "decoder", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving decoder model checkpoint to %s", output_dir)
        self.decoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class UnsupervisedModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(UnsupervisedModel, self).__init__()

        self.modeling_method = args.modeling_method
        if (self.modeling_method in ["VRAG", "VRAG_magical"]):
            self.kl_beta = args.kl_beta
        elif (self.modeling_method == "RL"):
            self.num_docs = args.num_docs

        self.topk = args.topk
        self.prior_model = PriorModel(args, indexed_passages)
        self.posterior_model = PosteriorModel(args, indexed_passages)
        self.decoder_model = DecoderModel(args)
        if (self.modeling_method == "VRAG_magical"):
            self.magical_model = MagicalModel(args, indexed_passages)
        self.multitask = args.multitask
        self.weigh_cannot_answer = args.weigh_cannot_answer
        if (self.weigh_cannot_answer):
            self.weight = args.weight
        self.fix_DPR = args.fix_DPR
        self.fix_prior = args.fix_prior
        self.fix_posterior = args.fix_posterior
        self.fix_decoder = args.fix_decoder
        self.args = args
        self.eps = 1e-10


    def SampleCategorical(self, dist, num_samples):
        samples = []
        for _ in range(num_samples):
            s = torch.distributions.categorical.Categorical(
                dist).sample().tolist()
            samples.append(s)
        samples = torch.tensor(samples).T.tolist()
        return samples

    def SelectByIndices(self, x, indices):
        l = []
        for i in range(len(indices)):
            l.append([])
            for j in range(len(indices[0])):
                l[-1].append(x[i][indices[i][j]])
        return l

    def SelectDoc(self, documents, indices):
        # documents: a list of B*K*(text)
        # indices: tensor (B,n)
        B, n = indices.size()
        results = []
        for i in range(B):
            result = []
            for j in range(n):
                result.append(documents[i][indices[i][j]])
            results.append(result)
        return results

    def forward(self, batch):
        prior_input_ids = batch['prior_input_ids']
        posterior_input_ids = batch['posterior_input_ids']
        posterior_token_type_ids = batch['posterior_token_type_ids']
        decoder_input_ids = batch['decoder_input_ids']
        decoder_response_ids = batch['decoder_response_ids']
        doc_ids = batch['doc_id']
        q_ids = batch['qid']
        has_cannot_answer = batch['has_cannot_answer']

        has_cannot_answer = has_cannot_answer.cuda()
        log_info = {}
        if (self.modeling_method == "RAG"):
            prior_logits, prior_indices, _ = self.prior_model(
                prior_input_ids.cuda(), self.topk)
            p_z_given_x = F.softmax(prior_logits, dim=-1) + EPSILON

            prior_indices = prior_indices.cpu().tolist()

            prior_topk_documents_text = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "text")
            prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")

            x = self.decoder_model._prepare_inputs(
                decoder_input_ids, decoder_response_ids, prior_topk_documents_text)

            if (self.multitask):
                decoder_input_ids_, decoder_response_ids_, _, cls_index = x
            else:
                decoder_input_ids_, decoder_response_ids_, _ = x

            if (self.multitask):
                decoder_loss, _, classification_logits = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda(), cls_index.cuda()])

                b_ = classification_logits.shape[0]
                k_ = classification_logits.shape[1]

                classification_logits = classification_logits.reshape(b_ * k_)

                has_cannot_answer_ = has_cannot_answer.repeat(
                    1, k_).reshape(b_ * k_)

                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                classification_loss = loss_fct(
                    classification_logits, has_cannot_answer_.float()).reshape(b_, k_)
                classification_loss = classification_loss.mean(dim=-1)
            else:
                decoder_loss, _ = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

            p_y_given_zx = torch.exp(-decoder_loss) + EPSILON
            p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + EPSILON
            loss = -torch.log(p_y_given_x)

            if (self.multitask):
                loss = loss * (1 - has_cannot_answer) + classification_loss

            if (self.weigh_cannot_answer):
                loss += (1 - has_cannot_answer) * (self.weight - 1) * loss
            loss = loss.mean()
        elif (self.modeling_method == "VRAG"):
            posterior_logits, posterior_indices, posterior_question_embeddings = self.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], self.topk)

            # ==================================================================
            # old implementation
            _, prior_indices, prior_question_embeddings = self.prior_model(
                prior_input_ids.cuda(), self.topk)

            posterior_indices = posterior_indices.cpu().tolist()
            prior_indices = prior_indices.cpu().tolist()

            posterior_topk_documents_text = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "text") # (B,K,L)
            posterior_topk_documents_ids = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "id")
            posterior_topk_documents_embeddings = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "embeddings")

            prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")
            prior_topk_documents_embeddings = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "embeddings")

            x = self.decoder_model._prepare_inputs(
                decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)

            if (self.multitask):
                decoder_input_ids_, decoder_response_ids_, _, cls_index = x
            else:
                decoder_input_ids_, decoder_response_ids_, _ = x

            if (self.multitask):
                decoder_loss, _, classification_logits = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda(), cls_index.cuda()])

                b_ = classification_logits.shape[0]
                k_ = classification_logits.shape[1]

                classification_logits = classification_logits.reshape(b_ * k_)

                has_cannot_answer = has_cannot_answer.cuda()
                has_cannot_answer_ = has_cannot_answer.repeat(
                    1, k_).reshape(b_ * k_)

                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                classification_loss = loss_fct(
                    classification_logits, has_cannot_answer_.float()).reshape(b_, k_)
                classification_loss = classification_loss.mean(dim=-1)
            else:
                decoder_loss, _ = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()]) # (B, K)

            posterior_dist = F.softmax(
                posterior_logits, dim=-1) + EPSILON # (B, K)

            if self.args.standard_mc:
                loss = decoder_loss.sum(dim=-1)/decoder_loss.size(1) # (B,)
            else:
                loss = (posterior_dist * decoder_loss).sum(dim=-1)

            if (self.multitask):
                loss = loss * (1 - has_cannot_answer) + classification_loss

            if (self.weigh_cannot_answer):
                loss += (1 - has_cannot_answer) * (self.weight - 1) * loss

            loss = loss.mean()

            prior_model_outputs = {
                "topk_documents_ids": prior_topk_documents_ids,
                "question_embeddings": prior_question_embeddings,
                "topk_documents_embeddings": prior_topk_documents_embeddings
            }
            posterior_model_outputs = {
                "topk_documents_ids": posterior_topk_documents_ids,
                "question_embeddings": posterior_question_embeddings,
                "topk_documents_embeddings": posterior_topk_documents_embeddings
            }

            KL = GetUnionKL(prior_model_outputs, posterior_model_outputs)
            loss += self.kl_beta * KL

        
            magical_indices = self.magical_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], self.topk)
            magical_indices = magical_indices.cpu()

            posterior_logits, posterior_indices, posterior_question_embeddings = self.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda(), magical_indices.cuda()], self.topk, magic=1)

            _, prior_indices, prior_question_embeddings = self.prior_model(
                [prior_input_ids.cuda(), magical_indices.cuda()], self.topk, magic=1)

            posterior_indices = posterior_indices.cpu().tolist()
            prior_indices = prior_indices.cpu().tolist()

            posterior_topk_documents_text = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "text")
            posterior_topk_documents_ids = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "id")
            posterior_topk_documents_embeddings = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "embeddings")

            prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")
            prior_topk_documents_embeddings = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "embeddings")

            x = self.decoder_model._prepare_inputs(
                decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)

            if (self.multitask):
                decoder_input_ids_, decoder_response_ids_, _, cls_index = x
            else:
                decoder_input_ids_, decoder_response_ids_, _ = x

            if (self.multitask):
                decoder_loss, _, classification_logits = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda(), cls_index.cuda()])

                b_ = classification_logits.shape[0]
                k_ = classification_logits.shape[1]

                classification_logits = classification_logits.reshape(b_ * k_)

                has_cannot_answer = has_cannot_answer.cuda()
                has_cannot_answer_ = has_cannot_answer.repeat(
                    1, k_).reshape(b_ * k_)

                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                classification_loss = loss_fct(
                    classification_logits, has_cannot_answer_.float()).reshape(b_, k_)
                classification_loss = classification_loss.mean(dim=-1)
            else:
                decoder_loss, _ = self.decoder_model(
                    [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

            posterior_dist = F.softmax(
                posterior_logits, dim=-1) + EPSILON

            loss = (posterior_dist * decoder_loss).sum(dim=-1)

            if (self.multitask):
                loss = loss * (1 - has_cannot_answer) + classification_loss

            if (self.weigh_cannot_answer):
                loss += (1 - has_cannot_answer) * (self.weight - 1) * loss

            loss = loss.mean()

            prior_model_outputs = {
                "topk_documents_ids": prior_topk_documents_ids,
                "question_embeddings": prior_question_embeddings,
                "topk_documents_embeddings": prior_topk_documents_embeddings
            }
            posterior_model_outputs = {
                "topk_documents_ids": posterior_topk_documents_ids,
                "question_embeddings": posterior_question_embeddings,
                "topk_documents_embeddings": posterior_topk_documents_embeddings
            }

            KL = GetUnionKL(prior_model_outputs, posterior_model_outputs)
            loss += self.kl_beta * KL
        elif self.modeling_method=="JSA":
            # retrieve documents by posterior model
            posterior_logits, posterior_indices, posterior_question_embeddings = self.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], self.topk)
            posterior_indices = posterior_indices.cpu().tolist()
            posterior_topk_documents_embeddings = torch.tensor(
                self.posterior_model.indexed_passages.get_field_by_indices(posterior_indices, "embeddings")).cuda() # (B, K, H)
            posterior_topk_documents_text = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "text") # a list of B*K*(text)
            posterior_topk_documents_ids = self.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "id")
            # retrieve documents by prior model
            _, prior_indices, prior_question_embeddings = self.prior_model(
                prior_input_ids.cuda(), self.topk)
            prior_indices = prior_indices.cpu().tolist()
            prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")

            # calculate topk recall
            batch_size = prior_input_ids.size(0)
            hidden_size = posterior_topk_documents_embeddings.size(-1)
            posterior_recall = 0
            prior_recall = 0
            for i in range(batch_size):
                if doc_ids[i] in posterior_topk_documents_ids[i]:
                    posterior_recall += 1
                if doc_ids[i] in prior_topk_documents_ids[i]:
                    prior_recall += 1
            posterior_recall /= batch_size
            prior_recall /= batch_size

            topk_probs = F.softmax(posterior_logits, dim=-1) # (B,K)
            prior_prob_on_topk = F.softmax(
                torch.bmm(posterior_topk_documents_embeddings, prior_question_embeddings.unsqueeze(2)).squeeze(2), dim=-1) # (B,K)
            # MIS step (only for not multitask)
            accept_rate = 0
            with torch.no_grad():
                # first sample independent z and feed them into the decoder
                z_ids = torch.multinomial(topk_probs, self.args.mis_step, replacement=True) # (B, n)
                final_z_ids = z_ids.clone() # n ids of z in the MIS chain
                doc_text = self.SelectDoc(posterior_topk_documents_text, z_ids)
                input_ids, labels, _ = self.decoder_model._prepare_inputs(decoder_input_ids, decoder_response_ids, doc_text) # (B, n, L)
                input_ids, labels = input_ids.view(-1, input_ids.size(-1)), labels.view(-1, labels.size(-1)) # (B*n, L)
                lm_probs, neg_log_lm_probs = self.decoder_model.cal_prob(input_ids.cuda(), labels.cuda()) # (B*n, )
                lm_probs, neg_log_lm_probs = lm_probs.view(z_ids.size(0), -1), neg_log_lm_probs.view(z_ids.size(0), -1) # (B, n)

                for i in range(self.args.mis_step):
                    # posterior prob
                    z_idx = z_ids[:, i].unsqueeze(1) # (B, 1)
                    posterior_prob = torch.gather(topk_probs, 1, z_idx).squeeze(1) #(B, )
                    # prior prob
                    prior_prob = torch.gather(prior_prob_on_topk, 1, z_idx).squeeze(1) # (B, ) 

                    # decoder prob
                    nl_lm_prob = neg_log_lm_probs[:, i] # (B, )
                    if i==0:
                        # compare with samples from last epoch
                        if self.args.jsa_cache and 'cached_ids' in batch:
                            cached_ids = batch['cached_ids'].cuda() # (B,1)
                            doc_idx = cached_ids.clone()
                            pv_prior_prob, pv_nl_lm_prob, pv_posterior_prob = batch['cached_probs'] # (B,)
                            pv_prior_prob, pv_nl_lm_prob, pv_posterior_prob = \
                                pv_prior_prob.cuda(), pv_nl_lm_prob.cuda(), pv_posterior_prob.cuda()
                            lm_prob_ratio = torch.exp(pv_nl_lm_prob-nl_lm_prob)
                            accept_probs = lm_prob_ratio*prior_prob*pv_posterior_prob/(pv_prior_prob*posterior_prob + self.eps)
                            for j in range(accept_probs.size(0)):
                                rand_num = random.random()
                                if rand_num<=accept_probs[j]:
                                    # accept
                                    accept_rate += 1
                                    doc_idx[j, :] = z_idx[j, :]

                        else:
                            # directly accept
                            doc_idx = z_idx.clone() # (B, 1)
                    else:
                        doc_idx = pv_doc_idx.clone()
                        lm_prob_ratio = torch.exp(pv_nl_lm_prob-nl_lm_prob)
                        accept_probs = lm_prob_ratio*prior_prob*pv_posterior_prob/(pv_prior_prob*posterior_prob + self.eps)
                        for j in range(accept_probs.size(0)):
                            rand_num = random.random()
                            if rand_num<=accept_probs[j]:
                                # accept
                                accept_rate += 1
                                doc_idx[j, :] = z_idx[j, :]
                    
                    final_z_ids[:, i] = doc_idx.squeeze(1)
                    
                    pv_posterior_prob = posterior_prob # previous posterior prob
                    pv_prior_prob = prior_prob
                    pv_nl_lm_prob = nl_lm_prob
                    pv_doc_idx = doc_idx
            
            # calculate prob with gradient
            if self.args.multi_sample_training:
                # use all the samples on the MIS chain
                doc_idx = final_z_ids # (B,1) --> (B,n)
            if self.args.in_batch_neg:
                # in batch negative sampling
                final_embeddings = torch.gather(
                    posterior_topk_documents_embeddings, 1, doc_idx.unsqueeze(-1).expand(batch_size, doc_idx.size(1), hidden_size)) # (B, n|1, H)
                final_embeddings = final_embeddings.view(-1, hidden_size) # (B*n, H)
                prior_similarities = prior_question_embeddings @ final_embeddings.T # (B, B*n) if not multi_sample_training then n=1
                prior_in_batch_probs = F.softmax(prior_similarities, -1)
                positive_idx = torch.arange(final_embeddings.size(0)).view(batch_size, -1).cuda() # (B, n)
                prior_prob = torch.gather(prior_in_batch_probs, 1, positive_idx).sum(-1) # (B, n)
                posterier_similarities = posterior_question_embeddings @ final_embeddings.T
                posterior_in_batch_probs = F.softmax(posterier_similarities, -1)
                posterior_prob = torch.gather(posterior_in_batch_probs, 1, positive_idx).sum(-1)
            else:
                # softmax in topk subspace
                prior_prob = torch.gather(prior_prob_on_topk, 1, doc_idx).sum(-1) # (B,)
                posterior_prob = topk_probs.gather(1, doc_idx).sum(-1)

            doc_text = self.SelectDoc(posterior_topk_documents_text, doc_idx)
            input_ids, labels, _ = self.decoder_model._prepare_inputs(decoder_input_ids, decoder_response_ids, doc_text) # (B, n|1, L)
            _, neg_log_lm_prob = self.decoder_model.cal_prob(
                input_ids.view(-1, input_ids.size(-1)).cuda(), labels.view(-1, labels.size(-1)).cuda()) # (B*n|1, )
            if self.args.multi_sample_training:
                neg_log_lm_prob = neg_log_lm_prob.view(batch_size, -1).sum(-1) # (B*n,)-->(B,n)-->(B,)
            log_prior_prob = torch.log(prior_prob+self.eps) # (B, )
            log_posterior_prob = torch.log(posterior_prob + self.eps) # (B, )
            loss = (-log_prior_prob + neg_log_lm_prob - log_posterior_prob).mean()
            log_info['accept_rate'] = accept_rate/((self.args.mis_step-1)*doc_idx.size(0))
            log_info['log_prior_prob'] = log_prior_prob.mean().item()
            log_info['log_posterior_prob'] = log_posterior_prob.mean().item()
            log_info['log_lm_prob'] = -neg_log_lm_prob.mean().item()
            log_info[f'prior_r{self.topk}'] = prior_recall
            log_info[f'posterior_r{self.topk}'] = posterior_recall
            if self.args.jsa_cache:
                batch['cached_ids'] = doc_idx.cpu()
                batch['cached_probs'] = (
                    prior_prob.detach().cpu(),
                    neg_log_lm_prob.detach().cpu(),
                    posterior_prob.detach().cpu()
                )


        return loss, log_info

    def ComputeMI(self, batch):
        (prior_input_ids,
         _,
         _,
         decoder_input_ids,
         decoder_response_ids,
         doc_ids,
         _,
         _) = batch

        prior_logits, prior_indices, _ = self.prior_model(
            prior_input_ids.cuda(), self.topk)
        p_z_given_x = F.softmax(prior_logits, dim=-1) + EPSILON

        prior_indices = prior_indices.cpu().tolist()
        prior_topk_documents_text = self.prior_model.indexed_passages.get_field_by_indices(
            prior_indices, "text")
        prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
            prior_indices, "id")

        x = self.decoder_model._prepare_inputs(
            decoder_input_ids, decoder_response_ids, prior_topk_documents_text)

        decoder_input_ids_, decoder_response_ids_, _ = x
        decoder_loss, _ = self.decoder_model(
            [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

        p_y_given_zx = torch.exp(-decoder_loss) + EPSILON
        p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + EPSILON

        prior_best_document_text = self.prior_model.indexed_passages.get_field_by_doc_id(
            [[_] for _ in doc_ids], "text")

        x = self.decoder_model._prepare_inputs(
            decoder_input_ids, decoder_response_ids, prior_best_document_text)

        decoder_input_ids_, decoder_response_ids_, _ = x
        decoder_loss, _ = self.decoder_model(
            [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

        p_y_given_zx = torch.exp(-decoder_loss) + EPSILON
        # p_y_given_zx = ReplaceNAN(p_y_given_zx)

        l = torch.log(p_y_given_zx / p_y_given_x + EPSILON)
        l = l.cpu().numpy()

        if (np.isnan(l).any()):
            return None

        return l

    def save_model(self, args, model_name):
        self.prior_model.save_model(args, model_name)
        self.posterior_model.save_model(args, model_name)
        self.decoder_model.save_model(args, model_name)

    def GetParameters(self):
        prior_params = list(self.prior_model.parameters())
        posterior_params = list(self.posterior_model.parameters())
        decoder_params = list(self.decoder_model.parameters())

        if (self.fix_DPR or (self.fix_prior and self.fix_posterior)):
            return decoder_params
        elif (self.fix_posterior and self.fix_decoder):
            return prior_params
        elif (not self.fix_prior and not self.fix_posterior and not self.fix_decoder):
            return prior_params + posterior_params + decoder_params
