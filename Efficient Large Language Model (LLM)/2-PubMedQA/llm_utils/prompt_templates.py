from .prompt_creators import create_prompt_with_examples, create_nemo_prompt_with_examples


#***********Sentiment Analysis*******************


_positive_review = '''\
Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. \
When I\'m in a good mood it makes me feel better. A bad mood just evaporates like sugar in the rain. \
This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. One of life\'s hidden gems. \
This is a desert isle CD in my book. Why she never made it big is just beyond me. \
Everytime I play this, no matter black, white, young, old, male, female EVERYBODY says one thing "Who was that singing ?"\
'''

_negative_review = '''\
Batteries died within a year ...: I bought this charger in Jul 2003 and it worked OK for a while. \
The design is nice and convenient. However, after about a year, the batteries would not hold a charge. \
Might as well just get alkaline disposables, or look elsewhere for a charger that comes with batteries that have better staying power.\
'''

def _sentiment_prompt_template_no_examples(review):
    return f'Review: {review} Question: What is the sentiment of the review, positive or negative?'

def sentiment_prompt_template(review):
    main_prompt = f'Review: {review} Question: What is the sentiment of the review, positive or negative?'
    conversation_examples = [
        (_sentiment_prompt_template_no_examples(_positive_review), 'positive'), 
        (_sentiment_prompt_template_no_examples(_negative_review), 'negative')
    ]
    return create_prompt_with_examples(main_prompt, conversation_examples=conversation_examples)


#***********List Generation*******************


def gen_list_template_nemo(len:int, thing:str) -> str:
    conversation_examples = [
        ('Make a python list of 2 animals.', '["dog", "spotted owl"]'),
        ('Make a python list of 3 books.', '["The Three Body Problem", "Dandelion Wine", "Deep Learning and the Game of Go"]'),
        ('Make a python list of 6 times of day.', '["morning", "evening", "night", "midday", "midnight", "dawn"]')
    ]
    main_prompt = f'Make a python list of {len} {thing}.'
    return create_nemo_prompt_with_examples(main_prompt, conversation_examples=conversation_examples)


def gen_list_template_zero_shot(n, topic):
    return f'Make a python list of {n} {topic}.'


#***********Customer Email Generation*******************


def customer_email_prompt_template(sender_name, company_name, industry, product, mood, store_location):
    return f"""\
Write a 150 word email from a customer named {sender_name} to the fictitious company {company_name} \
that ends in the customer's name.

Context: The mood of the customer {sender_name} is {mood}.

Instructions: Take the following steps in drafting this email from the customer {sender_name} to the {industry} company {company_name}:
1) The customer makes a question or complaint about the following product: {product}.
2) The customer tells a brief story about a relevant experience they had with their {company_name} {product}.
2) The customer describes that they purchased the product at a {company_name} store location in {store_location}.
3) The customer signs off in a way that matches their mood using their name {sender_name}.
4) If the customer has\'t signed off with their name ({sender_name}) they sign off with their name ({sender_name}).
"""


#***************Extractors*******************


def extract_template(extract_question, text):
    return f'{extract_question} {text} answer:'

def extract_name_template(text):
    return f'Who sent the email? {text} answer:'

def extract_product_template(text):
    return f'What product of StarBikes is this email about? {text} answer:'

def extract_location_template(text):
    return f'Where was the store located? {text} answer:'


#***********Autoresponse Email Generation*******************


def customer_response_email_prompt_template(company_name, recipient_name, sentiment, product, store_location):
    return f"""\
Write a customer response email of 200 words.

Context: {recipient_name} just sent an email expressing a {sentiment} sentiment about our company {company_name}. \
Their email was about a {product} they purchased in {store_location}. We want to send a response email emphathizing with \
their experience and if appropriate, telling them that someone from {store_location} will be contacting them soon.

Instructions: Write a reply email using the following steps:

1) Greet {recipient_name} professionally by their name. If their name is 'the customer' address them as "Dear Customer".

2) Depending on the sentiment expressed in their email (stated above), empathize as a friend would about their experience with their {product}.

3) Tell the customer that an employee from our store in {store_location} will contact them soon \
to followup with them in more detail. NEVER ask the customer to contact us. NEVER ask the customer for thier contact information.

4) Write a professional closing signed by "{company_name} Customer Support Team"
"""