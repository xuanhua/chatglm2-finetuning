"""
This file originally copied from below link:
https://github.com/liucongg/ChatGLM-Finetuning/blob/v0.1/train_pipeline.py         
"""
import os.path

import torch

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer

from deepspeed.pipe import PipelineModule, LayerSpec
from torch.nn import CrossEntropyLoss
import deepspeed
import argparse
import math
import time
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import random
import numpy as np
from peft import LoraConfig, get_peft_model

from config import CHATGLM_6B_V2_BASE_MODEL_PATH
from data_set import (
    GLMPromptDataSet,
    DataCollatorForPromptDataset,
    DataCollatorForDeepspeedPipelineModel
)

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

# There is no need for get_masks in finetuning of chatglm2_6b
#def get_masks(input_ids, device):
#    batch_size, seq_length = input_ids.shape
#    # Here 150004 is the bos_token_id, check configuration_chatglm.py of chatglm_6b project.
#    context_lengths = [seq.tolist().index(150004) for seq in input_ids]
#    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
#    attention_mask.tril_()
#    for i, context_length in enumerate(context_lengths):
#        attention_mask[i, :, :context_length] = 1
#    attention_mask.unsqueeze_(1)
#    attention_mask = (attention_mask < 0.5).bool()
#    return attention_mask

#def get_position_ids(input_ids, mask_positions, device):
#    batch_size, seq_length = input_ids.shape
#    # Here 150004 is the bos_token_id, check configuration_chatglm.py of chatglm_6b project.
#    context_lengths = [seq.tolist().index(150004) for seq in input_ids]
#    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
#    for i, context_length in enumerate(context_lengths):
#        position_ids[i, context_length:] = mask_positions[i]
#    block_position_ids = [torch.cat((torch.zeros(context_length, dtype=torch.long, device=device),
#                                     torch.arange(seq_length - context_length, dtype=torch.long,
#                                                  device=device) + 1
#                                     )) for context_length in context_lengths]
#    block_position_ids = torch.stack(block_position_ids, dim=0)
#    position_ids = torch.stack((position_ids, block_position_ids), dim=1)
#    return position_ids

class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.word_embeddings = model.transformer.embedding.word_embeddings
        self.weight = self.word_embeddings.weight
        self.rotary_pos_emb = model.transformer.rotary_pos_emb

    def forward(self, ipt):
        """
        Args:
            ipt (Tuple[torch.Tensor]): A input tuple, that includes `input_ids` and `labels`.
        
        Some terminologies:
            b: batch size
            s: sequence length
            h: hidden size
        """
        input_ids, labels = ipt
        hidden_states = self.word_embeddings(input_ids)

        # input_ids is in shape [b, s], where b is batch size, s is sequence length and h is hidden size
        seq_len = input_ids.size(1)

        rotary_pos_emb = self.rotary_pos_emb(seq_len)[None, :seq_len]
        # rotary_pos_emb is in shape [s, b, 32, 2]
        rotary_pos_emb = rotary_pos_emb.transpose(0,1).contiguous()

        # hidden_states is in shape [s, b, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states, rotary_pos_emb, labels

class GLMBlockPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration, layer_idx):
        super().__init__()
        self.layer = model.transformer.encoder.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt):
        hidden_states, rotary_pos_emb, labels = ipt
        layer_ret = self.layer(hidden_states, 
                                   attention_mask=None,
                                   rotary_pos_emb=rotary_pos_emb,
                                   kv_cache=None,
                                   use_cache=True)
        hidden_states, _ = layer_ret
        return hidden_states, rotary_pos_emb, labels

class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.final_layernorm = model.transformer.encoder.final_layernorm

    def forward(self, ipt):
        hidden_states, rotary_pos_emb, labels = ipt
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, labels

class LMPipeLayer(torch.nn.Module):
    """
    Layers that transform hidden states to logits for the language modeling task.
    """
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()
        self.output_layer = model.transformer.output_layer

    def forward(self, ipt):
        # hidden_states: [s, b, h]
        hidden_states, labels = ipt

        # Did the same thing as in chatglm2_6b/modeling_chatglm.py line 951: hidden_states = transformer_outputs[0]
        #hidden_states = hidden_states[0]

        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0,1).contiguous()

        return lm_logits, labels

class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: ChatGLMForConditionalGeneration):
        super().__init__()

    def forward(self, ipt):
        logits, labels = ipt
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1))
        return loss

def get_model(model):
    layers = [
        LayerSpec(EmbeddingPipeLayer, model=model),
        *[LayerSpec(GLMBlockPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.config.num_layers)],
        LayerSpec(FLNPipeLayer, model=model),
        LayerSpec(LMPipeLayer, model=model),
        LayerSpec(LossPipeLayer, model=model)
    ]
    return layers

def set_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_path", default="data/tb_0.json", type=str, help="")
    parser.add_argument("--model_name_or_path", default=CHATGLM_6B_V2_BASE_MODEL_PATH, type=str, help="")
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

    #print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    #print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)
    #print_rank_0("tokenizer.bos_token_id: {}".format(tokenizer.bos_token_id), args.global_rank)
    #print_rank_0("tokenizer.bos_token: {}".format(tokenizer.bos_token), args.global_rank)
    #print_rank_0("tokenizer.eop_token_id: {}".format(tokenizer.eop_token_id), args.global_rank)
    #print_rank_0("tokenizer.eop_token: {}".format(tokenizer.eop_token), args.global_rank)

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

    train_dataset = GLMPromptDataSet(data_path=args.train_path,
                                     tokenizer=tokenizer,
                                     max_len = args.max_len,
                                     max_src_len= args.max_src_len,
                                     skip_overlength_example=True,
                                     ignore_pad_token_for_loss=True
                                    )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=ds_config["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=DataCollatorForDeepspeedPipelineModel(),
                                  drop_last=True,
                                  num_workers=0)

    #train_dataset = GLMPromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    #data_collator = DataCollatorForPromptDataset()

    #g = torch.Generator()
    #train_dataloader = DataLoader(train_dataset,
    #                              collate_fn=data_collator,
    #                              shuffle=True,
    #                              drop_last=True,
    #                              batch_size=args.per_device_train_batch_size,
    #                              generator=g)

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

    all_loss = 0.0
    start = time.time()
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