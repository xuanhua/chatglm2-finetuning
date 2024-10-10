from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration 

from deepspeed.pipe import PipelineModule
import deepspeed
from peft import LoraConfig, get_peft_model

# Following imported model coming from file: /home/ubuntu/anaconda3/lib/python3.9/site-packages/peft/peft_model.py
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraModel
from peft.utils.save_and_load import get_peft_model_state_dict

import random
import numpy as np
import torch

from config import CHATGLM_6B_V2_BASE_MODEL_PATH

import argparse
import os
import sys

from typing import List, Dict, Tuple, Union, Optional
import re

from train.finetuning_lora_with_pipeline import get_model

def _get_submodules(model, key):
    """
    This function is copied from peft/utils/other.py
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

import torch.nn as nn

def restore_peft_model_from_pipemodule_checkpoint(lora_model: LoraModel, 
                                                  target_module_names: List[str],
                                                  pipeline_model: PipelineModule):
    """
    Restore the peft model from a pipeline module checkpoint.

    Args:
        peft_model (PeftModel): The Peft model to be restored.
        target_module_names (List[str]): names of modules, which has been replaced with lora modules.
        pipeline_model (PipelineModule): The Pipeline module from which the state dict will be extracted and loaded into the Peft model.
    Returns:
        None
    """
    def _collect_target_and_parent_tuples(model: nn.Module, target_module_names: List[str]):
        """
        Collects tuples of (parent_module, target_child_module, target_name) from a given model and target_module_names.
        """
        # Record tuples of (parent_module, target_child_module, target_name)
        pc_tuple_list = []

        # Extract all the parent module of lora related modules
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if isinstance(target_module_names, str):
                target_module_found = re.fullmatch(target_module_names, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in target_module_names)
            
            if target_module_found:
                parent, target, target_name = _get_submodules(model, key)
                pc_tuple_list.append((parent, target, target_name))
        return pc_tuple_list
    
    # Collect tuples of (parent_module, target_child_module, target_name) from the lora model
    lora_pc_tuples = _collect_target_and_parent_tuples(lora_model, target_module_names)
    # Collect tuples of  (parent_module, target_child_module, target_name) from the pipeline model
    ppmodel_pc_tuples = _collect_target_and_parent_tuples(pipeline_model, target_module_names)

    assert len(lora_pc_tuples) == len(ppmodel_pc_tuples), "The number of target modules in the lora model and pipeline model should be equal."

    # Do replacement for lora model with target modules from pipeline model
    for dest_tuple, source_tuple in zip(lora_pc_tuples, ppmodel_pc_tuples):
        dest_parent, dest_target, dest_target_name = dest_tuple
        src_parent, src_target, src_target_name = source_tuple
        src_target.to('cpu')
        setattr(dest_parent, dest_target_name, src_target)

def set_random_seed(seed: int):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model_name_or_path", 
                        default=CHATGLM_6B_V2_BASE_MODEL_PATH, 
                        type=str,
                        help="Directory to original huggingface pretrained model",
                        required=False)
    parser.add_argument("--checkpoint_path",
                        default="",
                        type=str,
                        help="Diretory where save the checkpoint of fine-tuned model",
                        required=True)
    parser.add_argument("--num_stages",
                        default=1,
                        type=int,
                        help="",
                        required=False)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="")
    return parser.parse_args()

def dummy_ds_config():
    """
    TODO: this ds_config actually is also used by training script, it should be placed in a common place for both scripts.
    """
    ds_config = {"train_micro_batch_size_per_gpu": 1,
                 "gradient_accumulation_steps": 1,
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
    return ds_config

def dump_lora_adapter_model_from_pipelinemodule(args):
    """
    Save the LoRA adapter model from PipelineModule to the same directory of PipelineModule's checkpoints.
    
    Note that, make sure you run this function under distributed mode with only GPU-0. Check below command:
    ```bash
    export CUDA_VISIBLE_DEVICES=0  
    deepspeed --master_port 5524 \
       ./predict_pipe_lora_new.py \
       --orig_model_name_or_path \
       /home/ubuntu/proj/chatglm-model-files-2023-06-12/ChatGLM-6B \
       --checkpoint_path \
       /data/xuanhua/chatglm-finetuned-models/output_dir_pipeline/global_step3000 \
       --num_stages \
       1
    ```
    """
    if args.local_rank == -1:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cpu", args.local_rank)
        deepspeed.init_distributed(dist_backend="gloo")

    args.global_rank = torch.distributed.get_rank()

    # This only work when env variable CUDA_VISIBLE_DEVICES is set to indicate only one GPU is available
    # otherwise there could be hang
    if args.global_rank != 0:
        print(f"Rank {args.global_rank} does not participate this these jobs, just exit with 0")
        sys.exit(0)

    set_random_seed(1234)
    model = ChatGLMForConditionalGeneration.from_pretrained(args.orig_model_name_or_path)

    lora_config = LoraConfig(r=8,
                             lora_alpha=32,
                             target_modules=["query_key_value"],
                             lora_dropout=0.1,
                             bias="none",
                             task_type="CASUAL_LM",
                             inference_mode=False
                            )
    # Trun it into lora model first
    peft_model = get_peft_model(model, lora_config)
    peft_model.to('cpu')
    # True lora model into Pipeline model

    layers=get_model(peft_model)
    print("Model layers: ", len(layers))

    model_pipe = PipelineModule(layers=layers, 
                                num_stages=args.num_stages)
    engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=dummy_ds_config(), model_parameters=model_pipe.parameters())

    # TODO: the complete solution should be:
    # 1. Compose an peft model from huggingface's transformer model
    # 2. Load the pipeline module by deepspeed's inference engine with the checkpoint path.
    # 3. Extract the lora adapter model's state dict M0 from the pipeline module.
    # 4. Convert M0 to the peft model's state dict M1. (Note that M0 could spread across multiple GPU or Nodes)
    # But now in our chatglm_v1_6B model, we use a single nvidia 3090 gpu to do above steps. So we could just use the peft model to save the lora adapter state dict directly.
    # Because in such case, both peft and pipeline module are on the same device and use the same adapter model phisically.
    #ds_engine._load_checkpoint(args.checkpoint_path)

    load_dir = os.path.dirname(args.checkpoint_path.rstrip("/")) 
    tag = os.path.basename( args.checkpoint_path.rstrip("/") )
    engine.load_checkpoint(load_dir=load_dir, 
                           tag=tag,
                           load_module_only=True)
    

    restore_peft_model_from_pipemodule_checkpoint( lora_model=peft_model.base_model,
                                                   target_module_names=["query_key_value"],
                                                   pipeline_model=engine.module)

    # Note that: this function only save the lora adapter itself to the `save_directory` not all model parameters
    peft_model.save_pretrained(save_directory=args.checkpoint_path)

if __name__ == "__main__":
    args = set_args()
    if not os.path.isdir(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory: {args.checkpoint_path} does not exist")
    dump_lora_adapter_model_from_pipelinemodule(args)