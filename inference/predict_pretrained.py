import torch
import json
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer

#from modeling_chatglm import ChatGLMForConditionalGeneration
#from tokenization_chatglm import ChatGLMTokenizer
#from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import time
import os
import argparse
from pprint import pprint
from logzero import logger

from transformers import AutoTokenizer



from config import CHATGLM_6B_V2_BASE_MODEL_PATH

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/tb_1.jsonl', type=str, help='')
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--model_dir',
                        default=CHATGLM_6B_V2_BASE_MODEL_PATH, 
                        type=str,
                        help='')
    parser.add_argument('--max_len', type=int, default=768, help='Maximum allowed length of tokens, including the prompt part and generated part')
    parser.add_argument('--max_src_len', type=int, default=450, help='Maximum allowed length of token in prompt part')
    parser.add_argument('--result_path', type=str, required=True, help='The path to save the predicted results' )
    return parser.parse_args()

def main():
    args = set_args()
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    #tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir, padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_V2_BASE_MODEL_PATH, trust_remote_code=True)
    model.eval()
    model.half().to("cuda:{}".format(args.device))

    save_data = []
    f1 = 0.0
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])

                if len(src_tokens) >= args.max_src_len:
                    logger.warning(f"{i}th src text is too long, skipping this line.")

                tokens = src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": 0.1,
                    "temperature": 0.7, # it must be positive
                    "do_sample": True,
                    "num_return_sequences": 1,
                }
                response = model.generate(input_ids, **generation_kwargs)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)
                pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                real_res = sample["answer"].split("\n")
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                save_data.append(
                    {"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": res[0], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print(f1 / 50)
    saved_path = args.result_path
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    with open(saved_path, "w", encoding="utf-8") as fin:
        pprint(save_data, fin)

if __name__ == '__main__':
    main()