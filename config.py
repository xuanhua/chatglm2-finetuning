import os

HOME = os.path.dirname(__file__) 
DATA_HOME = os.path.join(HOME, 'data')

# Model weights of original Chatglm-6b v1 model
#CHATGLM_6B_V1_MODEL_PATH="/home/ubuntu/proj/chatglm-model-files-2023-06-12/ChatGLM-6B"

# huggingface model downloaded
CHATGLM_6B_V2_BASE_MODEL_PATH="/data/xuanhua/hg_models/chatglm2_6b"
CHATGLM_6B_V1_BASE_MODEL_PATH="/data/xuanhua/hg_models/chatglm_6b"

#PROMPT_TEXT = "你现在是一个行程预定助理，你从用户和机器人的对话中，总结并生成Json结构的回复内容，包括‘行程’以及‘回复’ 两个部分。行程中会包括机票、火车票、酒店、用车等预定要求；下面是用户和机器人的对话："

# Model weights of lora fine-tuned Chatglm-6b v1 model
# More models please check directory /data/xuanhua/chatglm-finetuned-models/output_dir_pipeline
#CHATGLM_6B_V1_LORA_MODEL_PATH="/data/xuanhua/chatglm-finetuned-models/output_dir_pipeline/global_step3000"

# Model checkpoints saved path (diretory) 
#MODEL_CHECKPOINTS_DIR = "/data/xuanhua/chatglm-finetuned-models/output_dir_pipeline/"