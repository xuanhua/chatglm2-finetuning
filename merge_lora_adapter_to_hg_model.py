from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration

from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraModel
from peft.utils.save_and_load import get_peft_model_state_dict
import random
import numpy as np
import torch
#from config import CHATGLM_6B_V1_MODEL_PATH
#from config import CHATGLM_6B_V1_LORA_MODEL_PATH

import argparse
import os
from typing import List, Dict, Tuple, Union, Optional
import shutil

from config import CHATGLM_6B_V2_BASE_MODEL_PATH

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model_name_or_path", default=CHATGLM_6B_V2_BASE_MODEL_PATH, type=str, help="", required=False)
    parser.add_argument("--checkpoint_path", default="", type=str, help="directory where lora adapter models resides", required=True)
    parser.add_argument("--merged_model_path", default="", type=str, help="directory that merged model saved to", required=True)
    return parser.parse_args()

def save_lora_adapter_to_hg_model(args):
    """
    Save the LoRA adapter model to original huggingface model (transformer model).
    """
    if not os.path.isdir(args.merged_model_path):
        raise FileNotFoundError(f"As the output directory, merged_model_path = {args.merged_model_path}")
    
    if args.merged_model_path == args.orig_model_name_or_path:
        raise ValueError(f"merged_model_path cannot be the same with orig_model_name_or_path")

    # Load original huggingface model (transformer model).
    model = ChatGLMForConditionalGeneration.from_pretrained(args.orig_model_name_or_path)
    
    # Create Peft model from original huggingface model and LoRA adapter saved directory
    # by default, the lora layers are actually been merged to original huggingface model
    peft_model = PeftModel.from_pretrained(model, args.checkpoint_path)
    # Here perf_model.base_model actually is an instance of LoraModel
    merged_model = peft_model.base_model.merge_and_unload()
    merged_model.save_pretrained(args.merged_model_path, max_shard_size="2GB")

if __name__ == "__main__":
    args = set_args()
    save_lora_adapter_to_hg_model(args)