
import argparse
import os
from tqdm import tqdm
import torch
import json
from transformers import AutoTokenizer, AutoModel

from  utils import save_data_with_pprint

from config import CHATGLM_6B_V2_BASE_MODEL_PATH
from config import DATA_HOME

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_V2_BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_6B_V2_BASE_MODEL_PATH, trust_remote_code=True).half().cuda()
model = model.eval()

# We might use it later for setting up different models, testing data paths, and other parameters
#def set_args():
#  parser = argparse.ArgumentParser()
#  parser.add_argument('--test_path', default='data/tb_1.jsonl', type=str, help='')
#  parser.add_argument('--device', default='1', type=str, help='')
#  parser.add_argument('--max_len', type=int, default=768, help='Maximum allowed length of tokens, including the prompt part and generated part')
#  parser.add_argument('--max_src_len', type=int, default=450, help='Maximum allowed length of token in prompt part')
#  parser.add_argument('--result_path', type=str, required=True, help='The path to save the predicted results' )
#  return parser.parse_args()

result = []
input_path = os.path.join(DATA_HOME, "tb_1.jsonl") 
result_path = os.path.join(DATA_HOME, "output", "tb_1_result_chatglm2_hg_mode.txt") 
with open(input_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                response, history = model.chat(tokenizer, sample['text'], history=[])
                result.append(    {"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": response }) 

save_data_with_pprint(result, result_path)