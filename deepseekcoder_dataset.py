import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

import transformers

from typing import (
    Dict,
    Sequence
)
import copy

"""
In this implementation most of the codes are copied from the deepseek coder's github repo
"""

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

#def _build_prompt(instruction: str):
#    return '''
#You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
#### Instruction:
#{}
#### Response:
#'''.format(instruction.strip()).lstrip()

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # model_max_length is now passed in from commandline parser, check finetuning_deepseek_coder_with_
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer):
    assert len(examples) == 1
    sources = [
        build_instruction_prompt( example['instruction'])
        for example in examples
    ]
    targets = [f"{example['output']}\n{EOT_TOKEN}" for example in examples]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

class DeepseekCoderDataSet(Dataset):
    def __init__(self, 
                 data_path:str,
                 #tokenizer:ChatGLMTokenizer,
                 tokenizer: transformers.AutoTokenizer,
                 max_len:int,
                 max_src_len:int,
                 batch_size:int = 1,
                 skip_overlength_example:bool=True,
                 ignore_pad_token_for_loss:bool=True):
        """
        Args:
            data_path (str): path to the raw dataset file.
            tokenizer (ChatGLMTokenizer): tokenizer instance to use.
            max_len (int): maximum length of all tokens, including query, answer and paddings.
            batch_size (int): number of example in each batch.
            max_src_len (int): maximum length of source text.
            skip_overlength_example (bool): if True, examples that exceed the `max_len` will be skipped.
        """
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                example = json.loads(line.strip())
                # TODO: handle a batch of examples, now for quick testing, it is used 

                input_dict = train_tokenize_function([example], tokenizer)

                self.all_data.append(
                    {"input_ids": input_dict["input_ids"][0], 
                     "labels": input_dict["labels"][0]
                    })

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

class DataCollatorForDscPipelineModel(object):
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