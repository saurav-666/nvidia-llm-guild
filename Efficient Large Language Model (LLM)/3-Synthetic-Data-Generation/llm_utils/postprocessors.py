import ast

def strip(response):
    return response.strip()


def postprocess_list(list_str):
    try:
        literal_list = ast.literal_eval(list_str)
    except:
        literal_list = []

    return list(set(literal_list))


def get_safe_name(name):
    unsafe = {'no', 'I', 'hello', 'hi', 'greetings', 'StarBikes'}
    return 'a customer who didn\'t give us their name' if name.strip().lower() in unsafe else name.strip()


def get_safe_product(product):
    return 'product' if product.strip().lower() == 'no' else product.strip()


def get_safe_location(location):
    return 'their store\'s location' if location.strip().lower() == 'no' else location.strip()