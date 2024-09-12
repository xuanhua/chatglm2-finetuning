import random
import numpy as np
import torch
import argparse
import os
from typing import List, Dict, Tuple, Union, Optional
import shutil
from logzero import logger

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from config import CHATGLM_6B_V2_BASE_MODEL_PATH

"""
For the fine-tuned model, following files will be copied from corresponding huggingface model to create a usable model like original one
* config.json
  - corresponding config.json created by fine-tuned model will be dropped.
* configuration_chatglm.py
* modeling_chatglm.py
* MODEL_LICENSE
* quantization.py
* README.md
* tokenization_chatglm.py
* tokenizer_config.json
* tokenizer.model

Check following for more details

1. config.json, comparing transformer-4.33.0 with transformer-4.27.0, there are several new fields added:
```json
{
    "classifier_dropout": null, # New
    "pre_seq_len": null, # New
    "prefix_projection": false, # New
    "quantization_bit": 0, # New
    "vocab_size": 65024 # New 
    "transformers_version": "4.33.0" # Updated, originally "4.27.1"
}
```
It does not change any existing parameters in original config.json (transformers-4.27.1), so the original config.json (transformers-4.27.1) will be kept.

2. pytorch_model.bin.index.json
In original model, model size:
```json
{
    "total_size": 12487168064
}
```

In freeze based fine-tuned model, model size:
```json
{
    "total_size": 10447567936
}

We find that, except the `total_size` field, there is no difference in model architecture (model parameters) for these two models.
The total model size is checked by 
```bash
du -b pytorch_model-0000* | awk '{sum+=$1} END {print sum}'
```

So currently, we still decide to use the new generated pytorch_model.bin.index.json for the fine-tuned model.

3. generation_config.json
It seems does not save unique and important information, just ignore it now.

"""

HG_MODEL_REQUIRED_FILES = [
    "config.json",
    "configuration_chatglm.py",
    "modeling_chatglm.py",
    "MODEL_LICENSE",
    "quantization.py",
    "README.md",
    "tokenization_chatglm.py",
    "tokenizer_config.json",
    "tokenizer.model"
]

DROPPED_FILES_IN_FINETUNED_MODEL = [
    "generation_config.json"
]

import os

def _rename_to_backup_file(path)->None:
    """
    Rename file with a new name, so that its content could be kept for future reference.
    """
    basename = os.path.basename(path)
    dirname  = os.path.dirname(path)
    #root, suffix = os.path.splitext(basename)
    uplimit = 10
    while uplimit > 0:
        basename = basename + ".bak"
        if not os.path.exists(os.path.join(dirname, basename)):
            break
        uplimit -= 1

    if uplimit <= 0:
        raise ValueError("Cannot find a unique backup name.")

    new_path = os.path.join(dirname, basename)
    shutil.move(path, new_path)

def rm_unused_in_finetuned_model(args:argparse.Namespace):
    """
    Drop unnecessary files in finetuned model, some of these files are conflicted with original HuggingFace GLM model,
    """
    finetuned_model_path = args.finetuned_model_path
    for fname in DROPPED_FILES_IN_FINETUNED_MODEL:
        shutil.rmtree(os.path.join(finetuned_model_path, fname), ignore_errors=True)

def cp_hg_model_required_files(args:argparse.Namespace):
    """
    huggingface transformers requires some files to be present in the model directory for it to work properly.
    (files that .save_pretrained() does not provides)
    so we must copied from the original model directory to the our merged model directory.
    """
    hg_model_path = args.hg_model_path
    finetuned_path = args.finetuned_model_path
        
    for fname in HG_MODEL_REQUIRED_FILES:
        src_path = os.path.join(hg_model_path, fname)
        dest_path = os.path.join(finetuned_path, fname)
        if os.path.exists(src_path):
            if os.path.exists(dest_path):
                _rename_to_backup_file(dest_path)
            shutil.copyfile(src_path, dest_path)
        else:
            logger.warning(f"{fname} in huggingface base model cannot be found!")

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hg-model-path", default=CHATGLM_6B_V2_BASE_MODEL_PATH, type=str, help="", required=False)
    parser.add_argument("--finetuned-model-path", default="", type=str, help="fine-tuned model directory path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = set_args()
    rm_unused_in_finetuned_model(args)
    cp_hg_model_required_files(args)