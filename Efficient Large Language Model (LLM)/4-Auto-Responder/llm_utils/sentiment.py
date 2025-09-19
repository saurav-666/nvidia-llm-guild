from typing import Any
from .data_classes import PromptWithLabel

def strip_response(response: str) -> str:
    return response.strip()

def generate_prompt_with_label(prompt:str, label:str) -> PromptWithLabel:    
    return PromptWithLabel(prompt=prompt, label=label)