from typing import Any
from .data_classes import PromptWithLabel

def get_clean_prediction(prediction: str) -> str:
    if 'yes' in prediction.lower():
        return 'yes'
    elif 'no' in prediction.lower():
        return 'no'
    else:
        return 'maybe'

def strip_response(response: str) -> str:
    return response.strip()

def generate_prompt_and_answer(data: Any) -> PromptWithLabel:
    prompt = ""
    for index, context in enumerate(data['CONTEXTS']):
        section_label = data['LABELS'][index]
        prompt += f"{section_label}: {context}\n"
    
    question_text = data['QUESTION']
    prompt += f"QUESTION: {question_text}\n"
    prompt += f"ANSWER (yes|no|maybe): "

    label = data['final_decision']
    
    return PromptWithLabel(prompt=prompt, label=label)