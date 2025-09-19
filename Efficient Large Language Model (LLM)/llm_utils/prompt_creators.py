from typing import List
from .data_classes import PromptWithLabel

def create_prompt_with_examples(main_prompt: str,
                                     conversation_examples: List[PromptWithLabel] = []) -> str:

    full_prompt = ""

    for user_message, agent_response in conversation_examples:
        full_prompt += user_message + agent_response + '\n'
        
    full_prompt += main_prompt
    return full_prompt

def create_nemo_prompt_with_examples(main_prompt: str,
                                     conversation_examples: List[PromptWithLabel] = []) -> str:

    # Initialize an empty string for examples
    full_prompt = ""

    # Process each conversation example
    for i, (user_message, agent_response) in enumerate(conversation_examples):
        # Format each example
        if i == 0:
            # The first example doesn't include "User:"
            full_prompt += f"{user_message}\n\nAssistant:{agent_response}\n\n"
        else:
            # Subsequent examples include "User:"
            full_prompt += f"User:{user_message}\n\nAssistant:{agent_response}\n\n"

    # Prepend examples to the main prompt if any
    if conversation_examples:
        final_prompt = full_prompt + "User:" + main_prompt
    else:
        final_prompt = main_prompt

    return final_prompt

def create_llama_prompt_with_examples(main_prompt: str,
                                      conversation_examples: List[PromptWithLabel] = [],
                                      system_context="") -> str:
    
    # Return the main prompt if no system context or conversation examples are provided
    if not system_context and not conversation_examples:
        return main_prompt

    # Start with the initial part of the prompt including the system context, if provided
    full_prompt = f"<s>[INST] <<SYS>>{system_context}<</SYS>>\n" if system_context else "<s>[INST]\n"

    # Add each example from the conversation_examples to the prompt
    for user_message, agent_response in conversation_examples:
        full_prompt += f"{user_message} [/INST] {agent_response} </s><s>[INST]"

    # Add the main user prompt at the end
    full_prompt += f"{main_prompt} [/INST]"

    return full_prompt
