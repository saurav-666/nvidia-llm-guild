def gen_list_template(len:int, thing:str) -> str:
    return f'Make a python list of {len} {thing}.'

def sentiment_template(review:str):
    return f'Review: {review} Question: What was the sentiment of the review, positive or negative? The sentiment of the review is'

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

def response_prompt_template(company_name, recipient_name, sentiment, product, store_location):
    return f"""\
Write a customer response email of 200 words to {recipient_name}.

Context: {recipient_name} just sent an email expressing a {sentiment} sentiment about our company {company_name}. \
{recipient_name}'s email was about a {product} they purchased in {store_location}.

Instructions: Take the following steps in replying to {recipient_name}:
1) Greet {recipient_name} professionally by their name
2) If their email had a positive sentiment, thank them for their appreciation of their {product}.
3) If their email had a negative sentiment, apologize that they are not satisfied.
4) Tell {recipient_name} that we dearly value our customers in {store_location}.
5) Tell {recipient_name} that a customer service rep from our store in {store_location} will reach out to them soon \
to find out more about how we can be of service.
6) Write a professional closing signed by "{company_name} Customer Support Team"\
"""