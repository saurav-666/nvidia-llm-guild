from typing import List
from .data_classes import PromptWithLabel

def create_prompt_with_examples(main_prompt: str, conversation_examples: List[PromptWithLabel] = []) -> str:
    """
    Constructs a full prompt string by concatenating conversation examples followed by a main prompt,
    intended for use with Large Language Models (LLMs) that have not undergone instruction fine-tuning.

    This function is designed to prepare a prompt that combines several conversation examples with a
    main prompt for LLMs. The conversation examples are composed of user messages and agent responses,
    concatenated as they are, without requiring a specific format tailored to any model's instruction
    fine-tuning prompt template. This approach is particularly useful for interacting with LLMs in their
    generic form, allowing for a broad applicability across different models without the need for custom
    formatting to align with specific fine-tuned instructions.

    Parameters:
    - main_prompt (str): The primary prompt to which the user is expected to respond.
    - conversation_examples (List[PromptWithLabel], optional): A list of PromptWithLabel tuples,
      where each tuple contains a 'prompt' (user message) and a 'label' (agent response), representing
      the conversational context. Defaults to an empty list if not provided.

    Returns:
    - str: A single string combining all conversation examples followed by the main prompt, separated by
      newlines, suitable for generic use with various LLMs without specific instruction fine-tuning.

    Note:
    - The function is particularly useful for crafting prompts for generic LLMs, where providing a sequence
      of conversational context can enhance response quality without necessitating adherence to model-specific
      instruction formats.
    """
    full_prompt = ""

    for user_message, agent_response in conversation_examples:
        full_prompt += user_message + agent_response + '\n'
        
    full_prompt += main_prompt
    return full_prompt



def create_nemo_prompt_with_examples(main_prompt: str, conversation_examples: List[PromptWithLabel] = []) -> str:
    """
    Generates a formatted prompt string for NeMo GPT models, incorporating conversation examples
    according to a specific instruction fine-tuning prompt template.

    This function is tailored for creating prompts that align with the expectations of NeMo GPT models
    that have been fine-tuned with specific instruction sets. The conversation examples are formatted
    to follow the model's prompt template, with explicit "User:" and "Assistant:" labels to delineate
    the conversational roles. This structured formatting is crucial for ensuring the model correctly
    interprets the provided context and generates responses that are consistent with the fine-tuning
    instruction template.

    Parameters:
    - main_prompt (str): The primary question or statement directed to the model.
    - conversation_examples (List[PromptWithLabel], optional): A list of PromptWithLabel tuples,
      where each tuple contains a 'prompt' (user message) and a 'label' (agent response), representing
      the conversational context. Defaults to an empty list if not provided.

    Returns:
    - str: A formatted string that combines the conversation examples with the main prompt, using
      explicit "User:" and "Assistant:" labels for each part of the conversation, tailored for
      instruction fine-tuned NeMo GPT models.

    Note:
    - The precise formatting, especially the inclusion of "User:" and "Assistant:" labels, is essential
      for the model to process the prompt correctly, reflecting the importance of adhering to the specific
      prompt structure used during the model's fine-tuning phase.
    """
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
                                      system_context: str = "") -> str:
    """
    Constructs a prompt string for LLaMA-2 models using the chat instruction fine-tuning prompt template,
    optionally incorporating a system context at the beginning.

    This function is specifically designed to create prompts that conform to the structured requirements
    of LLaMA-2 models that have been fine-tuned for chat interactions. It allows for the inclusion of a
    system context, which is a brief description or setting information that helps guide the model's
    responses. The function then formats conversation examples and the main prompt according to the
    LLaMA-2 instruction fine-tuning template, ensuring that the generated prompt aligns with the model's
    expectations for structure and context delineation.

    Parameters:
    - main_prompt (str): The primary query or statement for the model to respond to.
    - conversation_examples (List[PromptWithLabel], optional): A list of tuples containing user messages
      and agent responses, formatted as conversation history. Defaults to an empty list if not provided.
    - system_context (str, optional): A brief description or context for the system to consider before
      generating a response. Defaults to an empty string if not provided.

    Returns:
    - str: A string formatted according to the LLaMA-2 chat instruction fine-tuning prompt template,
      combining any provided system context, conversation examples, and the main prompt.

    Note:
    - The function's formatting includes specific tags like "<s>[INST]", "<<SYS>>", and "[/INST]",
      essential for the model to understand the different parts of the prompt. This structured approach
      is crucial for leveraging the chat instruction fine-tuning of LLaMA-2 models, allowing for more
      nuanced and contextually aware interactions.
    """
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
