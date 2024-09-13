from typing import (
  Dict,
  List,
  Union
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


if __name__ == "__main__":
  cls = load_class_from_package("/data/xuanhua/chatglm2-finetuned-models/output_freeze/global_step_449", 
                          "modeling_chatglm.ChatGLMForConditionalGeneration")
  print(f"{cls}")
  print("Done")