import json
from torch.utils.data import Dataset
import torch

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer

class GLMPromptDataSet(Dataset):
    def __init__(self, 
                 data_path:str,
                 tokenizer:ChatGLMTokenizer,
                 max_len:int,
                 max_src_len:int,
                 skip_overlength_example:bool):
        self.all_data = []
        n_skipped_examples = 0
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
                if skip_overlength_example and skip_flag:
                    n_skipped_examples += 1
                    continue
                self.all_data.append(
                    {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)})
        print("the number of skipping data is {}, the proportion is {}".format(n_skipped_examples, n_skipped_examples / (
                len(self.all_data) + n_skipped_examples)))

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

if __name__ == "__main__":
    # test the dataset and collator
    pass