def gen_list_template(len:int, topic:str) -> str:
    return f'Make a python list of {len} {topic}.'

def sentiment_template(text:str) -> str:
    return f'REVIEW: {text}\nQUESTION: Was the overall sentiment of the following review positive or negative?\nSENTIMENT (positive|negative):'

def persona_template(text:str) -> str:
    return f'{text}'