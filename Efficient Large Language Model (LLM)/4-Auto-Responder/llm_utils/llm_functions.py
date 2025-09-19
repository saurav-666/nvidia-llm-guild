from .nemo_service_models import NemoServiceBaseModel
from .models import Models, LoraModels
from .prompt_creators import create_prompt_with_examples

from .prompt_templates import (
    sentiment_prompt_template, 
    gen_list_template_nemo, 
    gen_list_template_zero_shot, 
    customer_email_prompt_template,
    extract_name_template,
    extract_product_template,
    extract_location_template,
    customer_response_email_prompt_template
)

from .postprocessors import (
    strip,
    postprocess_list,
    get_safe_name,
    get_safe_product,
    get_safe_location
)

def make_llm_function(model, prompt_template_function, postprocessor=None):
    """
    Creates a function for interacting with a language model (LLM), utilizing a specified prompt
    template function and an optional postprocessing function.

    This higher-order function takes a model, a prompt template function, and an optional postprocessor.
    It returns a new function (`llm_function`) that, when called, generates a prompt using the provided
    prompt template function, sends this prompt to the model for generation, and optionally applies
    a postprocessing step to the model's response.

    Parameters:
    - model: The language model to be used for generating responses. This model should have a `generate`
      method that accepts a prompt string and keyword arguments.
    - prompt_template_function (callable): A function that takes any number of arguments and returns a
      string. This function is responsible for creating the prompt that will be passed to the model.
    - postprocessor (callable, optional): An optional function that takes the model's response as input
      and returns a processed version of the response. If not provided, the model's response is returned
      as is.

    Returns:
    - callable: A function (`llm_function`) that takes arbitrary positional and keyword arguments to
      generate a prompt using `prompt_template_function`, sends this prompt to the model, and applies
      any postprocessing if `postprocessor` is provided. The function signature of the returned `llm_function`
      is `(*prompt_template_args, **kwargs)` where `*prompt_template_args` are arguments for the
      `prompt_template_function` and `**kwargs` are forwarded to the model's `generate` method.

    Example Usage:
    ```python
    # Define a simple prompt template function
    def prompt_template(name):
        return f"Hello, {name}! How can I assist you today?"

    # Define a model with a .generate() method (pseudo-code)
    model = SomeLanguageModel()

    # Optionally, define a postprocessor function
    def postprocess(response):
        return response.strip()

    # Create the LLM function
    llm_function = make_llm_function(model, prompt_template, postprocessor=postprocess)

    # Use the created function
    response = llm_function("Alice", max_length=50)
    ```

    Note:
    - The `make_llm_function` allows for flexible interaction with different language models and
      customization of the input prompt and response handling, facilitating easy adjustments to
      workflow or model changes.
    """
    
    def llm_function(*prompt_template_args, **kwargs):

        prompt = prompt_template_function(*prompt_template_args)

        response = model.generate(prompt, **kwargs)

        if postprocessor:
            response = postprocessor(response)

        return response

    return llm_function


get_sentiment = make_llm_function(NemoServiceBaseModel(Models.gpt8b.value), 
                                  sentiment_prompt_template, 
                                  postprocessor=strip)

generate_list_43B = make_llm_function(NemoServiceBaseModel(LoraModels.gpt43b.value),
                                      gen_list_template_nemo, 
                                      postprocessor=postprocess_list)

generate_list_8B_lora = make_llm_function(NemoServiceBaseModel(LoraModels.gpt8b.value, customization_id='03f25d3b-715d-44cb-b682-61ef6f7df476'),
                                          gen_list_template_zero_shot, 
                                          postprocessor=postprocess_list)

generate_customer_email = make_llm_function(NemoServiceBaseModel(Models.gpt43b.value), 
                                            customer_email_prompt_template, 
                                            postprocessor=strip)

extract_name = make_llm_function(NemoServiceBaseModel('gpt20b', customization_id='5ea8dd76-ba7c-49ea-a0bf-c97fcf819a86'), 
                                 extract_name_template, 
                                 postprocessor=get_safe_name)

extract_product = make_llm_function(NemoServiceBaseModel('gpt20b', customization_id='5ea8dd76-ba7c-49ea-a0bf-c97fcf819a86'), 
                                    extract_product_template, 
                                    postprocessor=get_safe_product)

extract_location = make_llm_function(NemoServiceBaseModel('gpt20b', customization_id='5ea8dd76-ba7c-49ea-a0bf-c97fcf819a86'), 
                                     extract_location_template, 
                                     postprocessor=get_safe_location)

generate_customer_response_email = make_llm_function(NemoServiceBaseModel(Models.gpt43b.value), 
                                                     customer_response_email_prompt_template, 
                                                     postprocessor=strip)

def autorespond_to_customer(email):
    sentiment = get_sentiment(email, tokens_to_generate=1)
    name = extract_name(email)
    product = extract_product(email)
    location = extract_location(email)

    response = generate_customer_response_email('StarBikes', name, sentiment, product, location)
    return response