import argparse
import json
import logging
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from .dataset import KnowledgeWalker, UnsupervisedDataset
from .models import UnsupervisedModel
from .scorer import Metrics
import torch.distributed as dist
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def Evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=eval_dataset.collate_fn
    )

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    prior_metrics = Metrics()
    posterior_metrics = Metrics()
    with torch.no_grad():
        model.eval()

        for batch in epoch_iterator:
            (prior_input_ids,
             posterior_input_ids,
             posterior_token_type_ids,
             _,
             _,
             doc_ids,
             _,
             _) = batch

            _, prior_indices, _ = model.prior_model(prior_input_ids.cuda(), args.topk)
            prior_indices = prior_indices.cpu().tolist()
            prior_topk_documents_ids = model.prior_model.indexed_passages.get_field_by_indices(
                prior_indices, "id")
            
            _, posterior_indices, _ = model.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], args.topk)
            posterior_indices = posterior_indices.cpu().tolist()
            posterior_topk_documents_ids = model.posterior_model.indexed_passages.get_field_by_indices(
                posterior_indices, "id")

            doc_ids = doc_ids
            for i in range(prior_input_ids.size(0)):
                prior_metrics.update_selection(prior_topk_documents_ids[i], doc_ids[i])
                posterior_metrics.update_selection(posterior_topk_documents_ids[i], doc_ids[i])

    return prior_metrics.scores(), posterior_metrics.scores()


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
    if (args.eval_only and args.params_file == None):
        args.params_file = os.path.join(args.model_path, "prior", args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    set_seed(args)
    dist_url='tcp://localhost:13457'
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.n_gpus, rank=0)
    indexed_passages = KnowledgeWalker(args)

    args.batch_size = args.batch_size * args.n_gpus
    model = UnsupervisedModel(args, indexed_passages).cuda()
    tokenizers = {
        "prior_tokenizer": model.prior_model.tokenizer,
        "posterior_tokenizer": model.posterior_model.tokenizer,
        "decoder_tokenizer": model.decoder_model.tokenizer
    }
    unsupervised_eval_dataset = UnsupervisedDataset(
        args, tokenizers, labels_file=args.labels_file, split=args.eval_dataset)
    prior_results, posterior_results = Evaluate(args, unsupervised_eval_dataset, model)

    print("***** Piror eval results *****")
    for key in sorted(prior_results.keys()):
        print(key, str(prior_results[key]))
    print("***** Posterior eval results *****")
    for key in sorted(posterior_results.keys()):
        print(key, str(posterior_results[key]))



if (__name__ == "__main__"):
    main()
