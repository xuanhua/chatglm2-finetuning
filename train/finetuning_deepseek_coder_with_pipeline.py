import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
import argparse

import math
import random
import numpy as np
import transformers
import time
import os

from peft import (
    LoraConfig,
    get_peft_model
)

import deepspeed
from deepspeed.pipe import (
    PipelineModule
)

#from transformers import LlamaForCausalLM
#from transformers.models.llama.modeling_llama import (
#    LlamaForCausalLM,
#    LlamaConfig,
#    LlamaDecoderLayer,
#    LlamaRMSNorm,
#)

from torch.utils.data import (
    DataLoader,
    RandomSampler
)

from deepseekcoder_dataset import (
    DeepseekCoderDataSet,
    DataCollatorForDscPipelineModel
) 

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

from models.deepseek_coder_pipeline import get_pipeline_model

"""
For dataset and finetuning related details, we reference this implementation 
https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/finetune

Let's build a more general implementation.
"""

from config import DEEPSEEKCODER_1_3B_BASE_MODEL_PATH

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_path", default="data/tb_0.json", type=str, help="")
    parser.add_argument("--model_name_or_path", default=DEEPSEEKCODER_1_3B_BASE_MODEL_PATH, type=str, help="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="")

    parser.add_argument("--max_len", type=int, default=2048, help="")
    parser.add_argument("--model_max_length", 
                        type=int, 
                        default=512,
                        help="This argument actually is the max_src_len, it is used for tokenizing batch of inputs")
    parser.add_argument("--max_src_len", type=int, default=1024, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--output_dir", type=str, default='models/output_dir_pipeline', help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")

    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    parser.add_argument("--save_model_step", default=40, type=int, help="")
    parser.add_argument("--num_stages", default=2, type=int, help="")
    parser.add_argument("--checkpoint_dir_started_from", default="", type=str,
                        help="if not empty, the training states will be recovered from this checkpoint."
                             "And this argument is optional")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    args = set_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed(dist_backend="nccl")

    args.global_rank = torch.distributed.get_rank()

    ds_config = {"train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "optimizer": {
                     "type": "Adam",
                     "params": {
                         "lr": 2e-5,
                         "betas": [
                             0.9,
                             0.95
                         ],
                         "eps": 1e-8,
                         "weight_decay": 5e-4
                     }
                 },
                 "fp16": {
                     "enabled": True
                 },
                 "zero_optimization": {
                     "stage": 1,
                     "offload_optimizer": {
                         "device": "cpu",
                         "pin_memory": True
                     },
                     "allgather_partitions": True,
                     "allgather_bucket_size": 2e8,
                     "overlap_comm": True,
                     "reduce_scatter": True,
                     "reduce_bucket_size": 2e8,
                     "contiguous_gradients": True
                 },
                 "steps_per_print": 5
                 }

    set_random_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        # For sft padding left or right are both OK, since all padding tokens are ignored by cross-entropy by design
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
        # Following arguments will be passed to tokenizer, as told by the documentation of `transformers.AutoTokenizer.from_pretrained` 
        model_max_length=args.model_max_length
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.gradient_checkpointing_enable()

    # Lora's config
    lora_config = LoraConfig(r=8, # copied from finetuning_lora.py
                        lora_alpha=32,
                        #target_modules=["query_key_value"],
                        target_modules=["k_proj", "q_proj", "v_proj"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )
    # Turn it into lora model first
    model = get_peft_model(model, lora_config)

    # Turn lora model into Pipeline model
    model_pipe = PipelineModule(layers=get_pipeline_model(model),
                                num_stages=args.num_stages)

    model_pipe.to(device).half()

    train_dataset = DeepseekCoderDataSet(data_path=args.train_path,
                                     tokenizer=tokenizer,
                                     max_len = args.max_len,
                                     max_src_len= args.max_src_len,
                                     skip_overlength_example=True,
                                     ignore_pad_token_for_loss=True
                                    )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=ds_config["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=DataCollatorForDscPipelineModel(),
                                  drop_last=True,
                                  num_workers=0)

    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)
    print_rank_0("args.per_device_train_batch_size = {}".format(args.per_device_train_batch_size), args.global_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)

    train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
    engine, _, _, _ = deepspeed.initialize(model=model_pipe,
                                           config=ds_config,
                                           model_parameters=model_pipe.parameters(),
                                           training_data=train_dataset)

    # tag that is used to load/save checkpoints
    tag_num = 0
    if args.checkpoint_dir_started_from:
        ckpt_started_from = args.checkpoint_dir_started_from.rstrip("/")
        tag = os.path.basename(ckpt_started_from)
        # Suppose that tag is in format: global_stepX (here X is the number of training steps that have been done before).
        tag_num = int(tag[len("global_step"):])

        ckpt_dir = os.path.dirname(ckpt_started_from)
        engine.load_checkpoint(load_dir=ckpt_dir, tag=tag)
    
    for step in range(args.num_train_epochs * num_update_steps_per_epoch):
        loss = engine.train_batch(data_iter=train_dataloader)
        print_rank_0("step = {}, loss = {}".format(step, loss.item()), args.global_rank)
        if (step + 1) % args.save_model_step == 0:
            print(f"Saving at step {step}")
            tag_num += step + 1
            engine.save_checkpoint(args.output_dir,
                                   tag=f"global_step{tag_num}")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
