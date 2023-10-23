import argparse
import json
import logging
import os, shutil
import random
from argparse import Namespace
from threading import local

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from .data import write_preds
from .dataset import KnowledgeWalker, UnsupervisedDataset
from .models import UnsupervisedModel
from .scorer import Metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def Train(args, train_dataset, eval_dataset, test_dataset, model, writer=None):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    model.zero_grad()
    global_step = 0
    best_acc = 0
    num_times_best_acc = 0
    best_found = False
    # init_eval
    # if dist.get_rank()==0:
    #     results_val = Evaluate(args, eval_dataset, model)
    #     logger.info("***** val results *****")
    #     for key in sorted(results_val.keys()):
    #         logger.info("  %s = %s", key, str(results_val[key]))

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        local_steps = 0
        tr_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration") if dist.get_rank()==0 else train_dataloader
        skip_counter = 0

        for step, batch in enumerate(epoch_iterator):
            global_step += 1

            # with torch.autograd.detect_anomaly():
            loss, log_info = model(batch=batch)
            if (torch.isnan(loss).sum() >= 1) and dist.get_rank()==0:
                skip_counter += 1
                print("skipped =", skip_counter)
                continue

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if dist.get_rank()==0:
                writer.add_scalar('loss', loss.item(), global_step)
                for key, value in log_info.items():
                    writer.add_scalar(key, value, global_step)
            tr_loss += loss.item()

            if ((step + 1) % args.gradient_accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                local_steps += 1
                if dist.get_rank()==0:
                    epoch_iterator.set_postfix(Loss=tr_loss / (local_steps + 1))

            if (args.save_every != 0 and global_step % args.save_every == 0) and dist.get_rank()==0:
                model.module.save_model(args, "checkpoint-" + str(global_step))
        if dist.get_rank()==0:
            results_val = Evaluate(args, eval_dataset, model)

            logger.info("***** val results *****")
            for key in sorted(results_val.keys()):
                logger.info("  %s = %s", key, str(results_val[key]))

            if (results_val["r@1"] > best_acc):
                num_times_best_acc = 0
                model.module.save_model(args, "best")
                best_acc = results_val["r@1"]
                best_found = True
            else:
                num_times_best_acc += 1
                if (num_times_best_acc == args.stopping_criteria):
                    break
            model.module.save_model(args, "checkpoint-" + str(global_step))
    if dist.get_rank()==0:
        model.module.save_model(args, "")
        if (not best_found):
            model.module.save_model(args, "best")


def Evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=eval_dataset.collate_fn
    )

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    metrics = Metrics()

    d = {}
    with torch.no_grad():
        model.eval()

        for batch in epoch_iterator:
            (prior_input_ids,
             _,
             _,
             decoder_input_ids,
             _,
             doc_ids,
             q_ids,
             _) = batch

            prior_logits, prior_indices, _ = model.module.prior_model(
                prior_input_ids.cuda(), args.topk)

            prior_dist = F.softmax(prior_logits, dim=-1).cpu().tolist()[0]
            prior_indices = prior_indices.cpu().tolist()

            decoder_input_ids = decoder_input_ids[0]

            prior_topk_documents_ids = model.module.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")[0]

            prior_topk_documents_text = model.module.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "text")[0]

            doc_ids = doc_ids[0]
            q_ids = q_ids[0]

            metrics.update_selection(prior_topk_documents_ids, doc_ids)

            if (args.eval_only):
                output_text_from_1_doc = []

                for j in range(len(prior_topk_documents_text)):
                    document_text = prior_topk_documents_text[j]
                    output_text_from_1_doc.append(model.module.decoder_model.generate_from_1_doc(
                        args, decoder_input_ids, document_text))

                output_text_from_k_docs = model.module.decoder_model.generate_from_k_docs(
                    args, decoder_input_ids, prior_topk_documents_text, prior_dist)

                d[q_ids] = {
                    "prior_dist": prior_dist,
                    "topk_documents_ids": prior_topk_documents_ids,
                    "generated_response_from_1_doc": output_text_from_1_doc,
                    "generated_response_from_k_docs": output_text_from_k_docs
                }

        if (args.eval_only):
            write_preds(eval_dataset, args.output_file, d, skip_cannot_answer=args.skip_cannot_answer)

    results = metrics.scores()
    return results


def main_worker(local_rank, args):
    dist_url='tcp://localhost:13457'
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.n_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    if local_rank==0:
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
        log_dir = os.path.join(args.model_path, 'log')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
    else:
        tb_writer = None

    # load args from params file and update the args Namespace
    if (args.eval_only and args.params_file == None):
        args.params_file = os.path.join(args.model_path, "prior", args.checkpoint, "params.json")
    # logger.info("using params from " + args.params_file)

    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)
    if local_rank==0:
        for key in vars(args):
            logger.info(str(key) + " = " + str(vars(args)[key]))

    args.params = params  # used for saving checkpoints

    # Set seed
    set_seed(args)

    indexed_passages = KnowledgeWalker(args)

    args.batch_size = args.batch_size * args.n_gpus
    model = UnsupervisedModel(args, indexed_passages)
    tokenizers = {
        "prior_tokenizer": model.prior_model.tokenizer,
        "posterior_tokenizer": model.posterior_model.tokenizer,
        "decoder_tokenizer": model.decoder_model.tokenizer
    }
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if (not args.eval_only):
        unsupervised_train_dataset = UnsupervisedDataset(
            args, tokenizers, split="train")
        unsupervised_eval_dataset = UnsupervisedDataset(
            args, tokenizers, split="val")
        unsupervised_test_dataset = UnsupervisedDataset(
            args, tokenizers, split="test")

        Train(args, unsupervised_train_dataset, unsupervised_eval_dataset,
              unsupervised_test_dataset, model, tb_writer)
    else:
        if local_rank==0:
            unsupervised_eval_dataset = UnsupervisedDataset(
                args, tokenizers, labels_file=args.labels_file, split=args.eval_dataset)
            results = Evaluate(args, unsupervised_eval_dataset, model)

            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Saved checkpoint directory")
    parser.add_argument("--dataroot", type=str, help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str,
                        help="Path to knowledge file.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead.")
    parser.add_argument("--output_file", type=str, default="",
                        help="Predictions will be written to this file.")
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--prior_path", type=str)
    parser.add_argument("--posterior_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument(
        "--build_index", action="store_true", help="Build index")
    parser.add_argument("--index_path", type=str, help="Path of the index")
    parser.add_argument("--n_gpus", type=int, default=1, help="Num GPUS")
    parser.add_argument("--dialog", action="store_true", help="dialog setting")
    parser.add_argument("--save_every", type=int,
                        help="save every nth step", default=0)
    parser.add_argument("--multitask", action="store_true",
                        help="Use multitask decoder")
    parser.add_argument("--weight", type=int,
                        help="weight for CANNOTANSWER", default=5)
    parser.add_argument("--weigh_cannot_answer",
                        action="store_true", help="use weight parameter")
    parser.add_argument("--skip_cannot_answer",
                        action="store_true", help="skip CANNOTANSWER")
    parser.add_argument("--fix_DPR", action="store_true",
                        help="fix DPR model weights")
    parser.add_argument("--fix_prior", action="store_true",
                        help="fix prior model weights")
    parser.add_argument("--fix_posterior", action="store_true",
                        help="fix posterior model weights")
    parser.add_argument("--fix_decoder", action="store_true",
                        help="fix DPR model weights")
    parser.add_argument("--standard_mc", action="store_true",
                        help="standard monte carlo method")
    parser.add_argument("--mis_step", type=int,
                        help="MIS steps", default=10)
    parser.add_argument("--dist_url", type=str,
                        help="communication url for ddp training", default="tcp://localhost:13457")
    args = parser.parse_args()
    mp.spawn(main_worker, nprocs=args.n_gpus, args=(args,))



if (__name__ == "__main__"):
    main()
