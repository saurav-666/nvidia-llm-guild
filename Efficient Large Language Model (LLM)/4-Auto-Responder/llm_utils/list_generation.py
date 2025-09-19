from typing import Any
from .data_classes import PromptWithLabel
import ast

def get_good_gen_list_len(response: str) -> str:
    try:
        literal_list = ast.literal_eval(response.strip())
    except:
        return -1

    try:
        num_unique_items = len(set(literal_list))
    except:
        return -1
    return num_unique_items

def generate_prompt_with_label(prompt:str, label:str) -> PromptWithLabel:    
    return PromptWithLabel(prompt=prompt, label=label)