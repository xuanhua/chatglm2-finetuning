import os

HOME = os.path.dirname(__file__) 
DATA_HOME = os.path.join(HOME, 'data')

# huggingface model downloaded
# https://huggingface.co/THUDM/chatglm2-6b
# d2e2d91789248536a747d9ce60642a336444186c
CHATGLM_6B_V2_BASE_MODEL_PATH="/data/xuanhua/hg_models/chatglm2_6b"

# huggingface model version
# https://huggingface.co/THUDM/chatglm-6b
# commit: bf0f5cfb575eebebf9b655c5861177acfee03f16
CHATGLM_6B_V1_BASE_MODEL_PATH="/data/xuanhua/hg_models/chatglm_6b"

MODEL_SAVED_HOME = "/data/xuanhua/chatglm2-finetuned-models/"

DEEPSEEKCODER_1_3B_BASE_MODEL_PATH=""