import torch
import json

USE_CHATGLM_6B_V1 = False
if USE_CHATGLM_6B_V1:
    from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
    from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer
else:
    from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
    from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer

from tqdm import tqdm
import time
import os
import argparse
from pprint import pprint
from logzero import logger

from transformers import AutoTokenizer

from utils import (
    BatchIterator,
    batch_collate_fn_chatglm1,
    batch_collate_fn_chatglm2 
)

if USE_CHATGLM_6B_V1:
    from config import CHATGLM_6B_V1_BASE_MODEL_PATH
    base_model_path = CHATGLM_6B_V1_BASE_MODEL_PATH
else:
    from config import CHATGLM_6B_V2_BASE_MODEL_PATH
    base_model_path = CHATGLM_6B_V2_BASE_MODEL_PATH 

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/tb_1.jsonl', type=str, help='')
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--model_dir',
                        default=base_model_path, 
                        type=str,
                        help='')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size, default value is 2')
    parser.add_argument('--max_len', type=int, default=768, help='Maximum allowed length of tokens, including the prompt part and generated part')
    parser.add_argument('--max_src_len', type=int, default=450, help='Maximum allowed length of token in prompt part')
    parser.add_argument('--result_path', default='data/output/tb_1_result_chatglm2_batch_mode.txt', type=str, help='The path to save the predicted results' )
    return parser.parse_args()

def main():
    args = set_args()
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model.eval()
    model.half().to("cuda:{}".format(args.device))

    # Collect results
    save_data = []
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()

    # Load all example into memory
    all_examples = []
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for _, line in enumerate(tqdm(fh, desc="iter")):
            all_examples.append(json.loads(line.strip()))

    if USE_CHATGLM_6B_V1:
        batch_iterator = BatchIterator(all_examples,
                                    batch_size=args.batch_size,
                                    batch_collate_fn=batch_collate_fn_chatglm1)
    else:
        batch_iterator = BatchIterator(all_examples,
                                       batch_size=args.batch_size,
                                       batch_collate_fn=batch_collate_fn_chatglm2)

    for _, (text_batch, ans_batch) in enumerate(tqdm(batch_iterator, desc="iter")):
        with torch.no_grad():
            inputs = tokenizer(text_batch, padding=True, return_tensors='pt').to(f"cuda:{args.device}")
            generation_kwargs = {
                "min_length": 5,
                "max_new_tokens": max_tgt_len,
                "top_p": 0.1,
                "temperature": 0.7, # it must be positive
                "do_sample": True,
                "num_return_sequences": 1,
            }
            response = model.generate(**inputs, **generation_kwargs)

            # Collect results from batch
            for i_b in range(args.batch_size):
                start_offset = inputs["input_ids"].shape[1]
                # TODO: change 'num_return_sequences' to fix this part later
                #for i_r in range(generation_kwargs["num_return_sequences"]):
                outputs = response.tolist()[i_b][start_offset:]
                r = tokenizer.decode(outputs).replace("<eop>", "")
                save_data.append(
                    {"text": text_batch[i_b], "ori_answer": ans_batch[i_b], "gen_answer": r})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    saved_path = args.result_path
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    with open(saved_path, "w", encoding="utf-8") as fin:
        pprint(save_data, fin)

if __name__ == '__main__':
    main()