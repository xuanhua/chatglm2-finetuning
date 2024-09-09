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