from typing import (
  Any,
  Dict,
  List,
  Iterator,
  Union,
  Tuple
)
from pprint import pprint

def save_data_with_pprint(data:Union[str, List, Dict], saved_path:str):
  """
  Save data in various formats in pretty print format. So that it can be readable by humans.
  Args:
    data: Data to save. It could be a string, list or dictionary.
    saved_path: Path (in string) where the data will be saved.
  """
  with open(saved_path, "w", encoding="utf-8") as fin:
    pprint(data, fin)


import sys
import os
import importlib
def load_class_from_package(package_dir:str, full_class_name:str):
  """
  Load a class from a package given its full name and the directory of the package.

  Typical usage:
  ```python
  cls = load_class_from_package("/data/xuanhua/chatglm2-finetuned-models/output_freeze/global_step_449", 
                          "modeling_chatglm.ChatGLMForConditionalGeneration")
  ```

  If you want to debug code using this function, make sure you use 'python -m yourscript.py' to launch the script.
  Here is the vscode launch configuration:
  ```json
  {
      "name": "debug utils.py",
      "type": "debugpy",
      "request": "launch",
      "python": "/home/ubuntu/anaconda3/bin/python",
      "module": "chatglm2_finetuning.utils",
      "cwd": "${workspaceFolder}/../",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ```
  This configuration equivalent to run the script with python -m command. like:
  ```bash
  # 'chatglm2_finetuning' is the current project
  python -m chatglm2_finetuning.utils
  ```
  For more information about 'python -m', please see: https://stackoverflow.com/questions/7610001/what-is-the-purpose-of-the-m-switch

  TODO: maybe we need a bash script for testing the fine-tuned model, which is wrapped as a huggingface model.
  """
  if not os.path.isdir(package_dir):
    raise ImportError("The provided package directory does not exist: {0}".format(package_dir))
  package_path = os.path.abspath(package_dir)
  if (package_path not in sys.path) and (package_dir not in sys.path):
    sys.path.insert(0, package_path)
  
  # For relative imports to work in package, we need pass package name to the importlib.import_module function.
  # So we need to add the parent directory of the package to the python path.
  package_parent_path = os.path.dirname(package_path)
  if (package_parent_path not in sys.path):
    sys.path.insert(0, package_parent_path)
  
  try:
    package_name = os.path.basename(package_dir)
    module_name, class_name = full_class_name.rsplit('.', maxsplit=1)
    module = importlib.import_module(package_name + "." + module_name)
    try:
      cls = getattr(module, class_name)
    except AttributeError as e:
      raise ImportError("Class does not exist: {0}".format(full_class_name)) from e
  finally:
    sys.path.pop(0)
  return cls

import math
class BatchIterator:
  def __init__(self, 
               all_examples:List[Any],
               batch_size:int,
               batch_collate_fn:callable=None) -> None:

    self._all_batches = []
    self._batch_size = batch_size
    self._num_batches = math.ceil(len(all_examples) / batch_size)
    self._cur_batch_index = 0
    self._batch_collate_fn = batch_collate_fn if batch_collate_fn else lambda x : x # identity function by default.

    for i in range(0, len(all_examples), batch_size):
      self._all_batches.append(all_examples[i : i + batch_size])
  
  def __iter__(self) -> Iterator[List[Any]]:
    return self

  def __next__(self) -> Any:
    if self._cur_batch_index >= self._num_batches:
      self._cur_batch_index  = 0
      raise StopIteration()
    else:
      batch  = self._all_batches[self._cur_batch_index]
      self._cur_batch_index  +=1
      return self._batch_collate_fn(batch)
  
  def __len__(self) -> int:
    return self._num_batches

def batch_collate_fn_chatglm1(example_batch:List[Any])->Tuple[Any, Any]:
  """
  Convert a list of examples into a tuple (text_array, answer_array) for chatglm 6b model
  """
  text_array, answer_array = [],[]
  for example in example_batch:
    text_array.append(example["text"])
    answer_array.append(example["answer"])
  return text_array, answer_array

def batch_collate_fn_chatglm2(example_batch:List[Any])->Tuple[Any, Any]:
  """
  Convert a list of examples into a tuple (text_array, answer_array) for chatglm2 6b model
  """
  text_array, answer_array = [], []
  for example in example_batch:
    text_array.append("[Round {}]\n\n问：{}\n\n答：".format(1, example["text"]))
    answer_array.append(example["answer"])
  return text_array, answer_array

if __name__ == "__main__":
  #cls = load_class_from_package("/data/xuanhua/chatglm2-finetuned-models/output_freeze/global_step_449", 
  #                        "modeling_chatglm.ChatGLMForConditionalGeneration")
  #print(f"{cls}")
  #print("Done")

  iter = BatchIterator([1,2,3,4,5], 2)
  for batch in iter:
    print(f"Batch: {batch}")

  iter = BatchIterator([{"text": "Hello", "answer":"Hi"}, {"text": "How are you?", "answer":"I'm fine"}], 3, batch_collate_fn)
  for batch in iter:
    print(f"Batch: {batch}")