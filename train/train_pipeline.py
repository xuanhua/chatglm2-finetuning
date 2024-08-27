"""
This file originally copied from below link:
https://github.com/liucongg/ChatGLM-Finetuning/blob/v0.1/train_pipeline.py         
"""
import os.path

import torch

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer

from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
from torch.nn import CrossEntropyLoss
import deepspeed
import argparse
import math
import json
import time
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import SchedulerType, default_data_collator, get_scheduler
from torch.utils.data import Dataset
import random
import numpy as np
from peft import LoraConfig, get_peft_model

from config import CHATGLM_6B_V2_BASE_MODEL_PATH

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def get_masks(input_ids, device):
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(150004) for seq in input_ids]
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(input_ids, mask_positions, device):
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(150004) for seq in input_ids]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    for i, context_length in enumerate(context_lengths):
        position_ids[i, context_length:] = mask_positions[i]
    block_position_ids = [torch.cat((torch.zeros(context_length, dtype=torch.long, device=device),
                                     torch.arange(seq_length - context_length, dtype=torch.long,
                                                  device=device) + 1
                                     )) for context_length in context_lengths]
    block_position_ids = torch.stack(block_position_ids, dim=0)
    position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    return position_ids


class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.word_embeddings = model.transformer.word_embeddings
        self.weight = self.word_embeddings.weight

    def forward(self, ipt):
        input_ids, labels = ipt
        hidden_states = self.word_embeddings(input_ids)
        mask_token = 150001
        seqs = input_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]
        attention_mask = get_masks(input_ids, device=input_ids.device)
        position_ids = get_position_ids(input_ids, device=input_ids.device, mask_positions=mask_positions)
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states, position_ids, attention_mask, labels


class GLMBlockPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration, layer_idx):
        super().__init__()
        self.layer = model.transformer.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt):
        hidden_states, position_ids, attention_mask, labels = ipt
        hidden_states = self.layer(hidden_states, position_ids, attention_mask, torch.tensor(self.layer_idx))[0]
        return hidden_states, position_ids, attention_mask, labels


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.final_layernorm = model.transformer.final_layernorm

    def forward(self, ipt):
        hidden_states, position_ids, attention_mask, labels = ipt
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, labels


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.word_embeddings = model.transformer.word_embeddings
        self.weight = self.word_embeddings.weight

    def forward(self, ipt):
        hidden_states, labels = ipt
        logits = torch.nn.functional.linear(hidden_states, self.word_embeddings.weight)
        logits = logits.permute(1, 0, 2).contiguous()
        return logits, labels


class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()

    def forward(self, ipt):
        logits, labels = ipt
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss


def get_model(model):
    layers = [TiedLayerSpec("word_embeddings", EmbeddingPipeLayer, model=model),
              *[LayerSpec(GLMBlockPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.config.num_layers)],
              LayerSpec(FLNPipeLayer, model=model),
              TiedLayerSpec("word_embeddings", LMPipeLayer, model=model),
              LayerSpec(LossPipeLayer, model=model)]
    return layers


def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_path", default="data/tb_0.json", type=str, help="")
    parser.add_argument("--model_name_or_path", default=CHATGLM_6B_V2_BASE_MODEL_PATH, type=str, help="", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="")

    parser.add_argument("--max_len", type=int, default=2048, help="")
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


class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        #prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
        prompt_text = "你现在是一个行程预定助理，你从用户和机器人的对话中，总结并生成Json结构的回复内容，包括‘行程’以及‘回复’ 两个部分。行程中会包括机票、火车票、酒店、用车等预定要求；下面是用户和机器人的对话："
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                prompt_tokens = tokenizer.tokenize(prompt_text)
                src_tokens = tokenizer.tokenize(sample["text"])
                src_tokens = prompt_tokens + src_tokens

                if len(src_tokens) > max_src_len:
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 3 - len(src_tokens)
                tgt_tokens = tokenizer.tokenize(sample["answer"])

                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                tokens = src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                assert len(tokens) <= max_len

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

                assert len(input_ids) == len(labels)
                assert len(input_ids) == max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append(
                    {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)})
        print("the number of skipping data is {}, the proportion is {}".format(skip_data_number, skip_data_number / (
                len(self.all_data) + skip_data_number)))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


class DataCollatorForPromptDataset(object):
    """Collate for supervised fine-tuning."""

    def __call__(self, samples):
        input_ids_list, labels_list = [], []
        for instance in samples:
            input_ids_list.append(instance["input_ids"])
            labels_list.append(instance["labels"])
        return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))


def collect_fn_glm(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["labels"])
    return ((pad_sequence(input_ids_list, batch_first=True), pad_sequence(labels_list, batch_first=True)),
            pad_sequence(labels_list, batch_first=True))


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

    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)

    print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)
    print_rank_0("tokenizer.bos_token_id: {}".format(tokenizer.bos_token_id), args.global_rank)
    print_rank_0("tokenizer.bos_token: {}".format(tokenizer.bos_token), args.global_rank)
    print_rank_0("tokenizer.eop_token_id: {}".format(tokenizer.eop_token_id), args.global_rank)
    print_rank_0("tokenizer.eop_token: {}".format(tokenizer.eop_token), args.global_rank)

    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.gradient_checkpointing_enable()

    # Lora's config
    lora_config = LoraConfig(r=8, # copied from finetuning_lora.py
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )
    # Turn it into lora model first
    model = get_peft_model(model, lora_config)

    # Turn lora model into Pipeline model
    model_pipe = PipelineModule(layers=get_model(model), num_stages=args.num_stages)

    model_pipe.to(device).half()

    train_dataset = GLMPromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    data_collator = DataCollatorForPromptDataset()

    g = torch.Generator()
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=args.per_device_train_batch_size,
                                  generator=g)

    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)
    print_rank_0("args.per_device_train_batch_size = {}".format(args.per_device_train_batch_size), args.global_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)

    train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
    engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=ds_config, model_parameters=model_pipe.parameters())

    # tag that is used to load/save checkpoints
    tag_num = 0
    if args.checkpoint_dir_started_from:
        ckpt_started_from = args.checkpoint_dir_started_from.rstrip("/")
        tag = os.path.basename(ckpt_started_from)
        # Suppose that tag is in format: global_stepX (here X is the number of training steps that have been done before).
        tag_num = int(tag[len("global_step"):])

        ckpt_dir = os.path.dirname(ckpt_started_from)
        engine.load_checkpoint(load_dir=ckpt_dir, tag=tag)

    start = time.time()
    all_loss = 0.0
    for step in range(args.num_train_epochs * num_update_steps_per_epoch):
        loss = engine.train_batch(data_iter=train_dataloader)
        print_rank_0("step = {}, loss = {}".format(step, loss.item()), args.global_rank)
        #all_loss += loss.item()
        #if args.local_rank == 0:
        #    if (step + 1) % args.show_loss_step == 0:
        #        now = time.time()
        #        avg_time = (now - start) / args.show_loss_step
        #        avg_loss = all_loss / args.show_loss_step
        #        print(f"Step={step:>6}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
        #        start = now
        #        all_loss = 0.0

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