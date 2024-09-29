import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer


def _build_prompt(query:str) -> str:
    """
    The origininal implementation is ChatGLMTokenizer.build_prompt(), but we did not use chat history in our implementation.
    This function is used to build prompt adapted to original ChatGLM2 model.
    
    Args:
        query (str): raw prompt string.
    """
    return  "[Round {}]\n\n问：{}\n\n答：".format(1, query)

class GLMPromptDataSet(Dataset):
    def __init__(self, 
                 data_path:str,
                 tokenizer:ChatGLMTokenizer,
                 max_len:int,
                 max_src_len:int,
                 skip_overlength_example:bool=True,
                 ignore_pad_token_for_loss:bool=True):
        """
        Args:
            data_path (str): path to the raw dataset file.
            tokenizer (ChatGLMTokenizer): tokenizer instance to use.
            max_len (int): maximum length of all tokens, including query, answer and paddings.
            max_src_len (int): maximum length of source text.
            skip_overlength_example (bool): if True, examples that exceed the `max_len` will be skipped.
        """
        self.all_data = []
        n_skipped_examples = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())

                # **Convert both query and answer part to ids with some special tokens in query part**
                # Below logic is copied from: git@github.com:THUDM/ChatGLM2-6B.git/ptuning/main.py
                prompt = _build_prompt(sample["text"])
                a_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_src_len)
                # The max_target_lenght = max_len - max_src_len - 1 (the last one is the special token 'eos')
                b_ids = tokenizer.encode(sample["answer"], add_special_tokens=False, truncation=True, max_length=max_len - max_src_len -1)

                # **Merge the query and answer part into one sequence and create two fields: 'inputs_ids', 'labels'**
                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.pad_token_id] * context_length + b_ids  + [tokenizer.eos_token_id]

                pad_len = max_len - len(input_ids)
                if pad_len < 0 and skip_overlength_example:
                    print(f"Ignore example {i}, query={prompt[:20]}...")
                    continue

                # **Padding the right side of the inputs_ids and labels with 'pad_token' ids**
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len 

                if ignore_pad_token_for_loss:
                    """
                    Set all src input ids as -100, which is ignored by loss function during training.
                    And also right side padding ids
                    """
                    labels = [ (l if l != tokenizer.pad_token_id else -100) for l in labels] 

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
        # Currently there should be no expanding needed, otherwise it should be wrong.
        if input_ids_list:
            for i in range(len(input_ids_list)):
                if i + 1 < len(input_ids_list):
                    assert len(input_ids_list[i]) == len(input_ids_list[i + 1]), "All instances should have the same length."
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100)
        }

class DataCollatorForDeepspeedPipelineModel(object):
    """Collate for deepspeed pipeline based model

    This collate function looks wierd because it's designed to work with DeepSpeed pipeline parallelism. 
    For different stags, it might require different input.

    In this collate function, it prepares inputs for both first and last stage of pipeline model:
    - For the first stage (stage 0), it only needs `input_ids` and `label` as inputs.
    - For the last stage (stage N-1), it needs `label`.

    So the output of this collate function includes required inputs for both kinds of stages in pipeline model.
    (The stage number is equal the number of used GPUs in usual)
    """
    def __call__(self, samples):
        input_ids_list, labels_list = [], []
        for instance in samples:
            input_ids_list.append(instance["input_ids"])
            labels_list.append(instance["labels"])
        return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))

if __name__ == "__main__":
    # test the dataset and collator
    pass