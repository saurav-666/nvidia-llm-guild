import os
import json
import ast
import textwrap
from functools import partial

from nemollm.api import NemoLLM

from .nemo_service_models import NemoServiceBaseModel
from .models import Models, LoraModels
from .helpers import sprint
from .prompt_templates import gen_list_template, customer_email_prompt_template, response_prompt_template
from .prompt_creators import create_nemo_prompt_with_examples, create_prompt_with_examples
from .llm_functions import make_llm_function


api_key = os.getenv('NGC_API_KEY')


def _initialize_respond_to_email():
    company_name = 'ByteWave'
    
    sentiment_model = NemoServiceBaseModel(Models.gpt8b.value, api_key=api_key, create_prompt_with_examples=create_prompt_with_examples)
    extractor_model = NemoServiceBaseModel('gpt20b', api_key=api_key, customization_id='5ea8dd76-ba7c-49ea-a0bf-c97fcf819a86')
    email_response_model = NemoServiceBaseModel(Models.gpt43b.value, api_key=api_key)
    pirate_model = NemoServiceBaseModel(LoraModels.gpt8b.value, api_key=api_key, customization_id='2d03bab3-0d79-44ff-b21f-bb6cea86b4fc')
    
    
    with open('4-Auto-Responder/data/solution_emails.json', 'r') as f:
        emails = json.load(f)
    
    
    def sentiment_prompt_template_no_examples(email):
        return f'Email: {email} Question: What was the sentiment of the email, positive or negative? The sentiment of the email is'
    
    def sentiment_prompt_template(email):
        main_prompt = f'Email: {email} Question: What was the sentiment of the email, positive or negative? The sentiment of the email is'
        conversation_examples = [
            (sentiment_prompt_template_no_examples(emails[10]), 'negative'),
            (sentiment_prompt_template_no_examples(emails[12]), 'positive')
        ]
        return sentiment_model.create_prompt_with_examples(main_prompt, conversation_examples=conversation_examples)
    
    
    def extract_template(extract_question, text):
        return f'{extract_question} {text} answer:'

    
    def safe_location(location):
        return 'their store\'s location' if location == 'no' else location
    
    def safe_product(product):
        return 'product' if product == 'no' else product
    
    def safe_name(name):
        unsafe = {'no', 'I', 'hello', 'hi', 'greetings', company_name}
        return 'the customer' if name.lower() is unsafe else name
    
    def pirate_template(statement):
        return f'Statement: {statement}Pirate Statement: '
    
    get_sentiment = make_llm_function(sentiment_model, sentiment_prompt_template)
    write_response_email = make_llm_function(email_response_model, response_prompt_template)
    pirate = make_llm_function(pirate_model, pirate_template)

    extractor = make_llm_function(extractor_model, extract_template)

    extract_name_prompt = f'Who sent the email?'
    extract_location_prompt = 'Where was the store located?'
    extract_product_prompt = f'What product of {company_name} is this email about?'


    safe_name_extractor = lambda extract_name_prompt, email: safe_name(extractor(extract_name_prompt, email))
    name_extractor = partial(safe_name_extractor, extract_name_prompt)

    safe_product_extractor = lambda extract_product_prompt, email: safe_product(extractor(extract_product_prompt, email))
    product_extractor = partial(safe_product_extractor, extract_product_prompt)

    safe_location_extractor = lambda extract_location_prompt, email: safe_location(extractor(extract_location_prompt, email))
    location_extractor = partial(safe_location_extractor, extract_location_prompt)
    
    
    def _respond_to_email(email):
        opening = 'Generating response to customer email:'
        underline = '-'*len(opening)
        print(f'{underline}\n{opening}\n{underline}\n{email}')
        
        sentiment = get_sentiment(email, tokens_to_generate=1)
        analysing_sentiment = 'Analysing email sentiment with LoRA fine-tuned NeMo GPT8B.'
        underline = '-'*len(analysing_sentiment)
        print(f'{underline}\n{analysing_sentiment}\n{underline}\n')
        print(f'Sentiment: {sentiment}\n')

        name = name_extractor(email)
        extracting_name = 'Extracting customer name, relevant product, and store location with LoRA fine-tuned NeMo GPT8B.'
        underline = '-'*len(extracting_name)
        print(f'{underline}\n{extracting_name}\n{underline}\n')
        print(f'Customer name: {name}')
        
        product = product_extractor(email)
        underline = '-'*len(extracting_name)
        print(f'Product: {product}')
        
        location = location_extractor(email)
        print(f'Store location: {location}\n')

        composing = 'Composing response with prompt engineered NeMo GPT43B.'
        underline = '-'*len(composing)
        print(f'{underline}\n{composing}\n{underline}\n')

        response = write_response_email(company_name, name, sentiment, product, location)
        underline = '-'*len(response)
        print(response+'\n')

        mimicking = 'Mimicking personal tone of voice with LoRA fine-tuned GPT8B.'
        underline = '-'*len(mimicking)
        print(f'{underline}\n{mimicking}\n{underline}\n')
        
        pirate_response = pirate(response)
        print(pirate_response)

    return _respond_to_email

respond_to_email = _initialize_respond_to_email()