def make_llm_function(model, prompt_template_function):
    
    def llm_function(*prompt_template_args, **kwargs):
        prompt = prompt_template_function(*prompt_template_args)
        response = model.generate(prompt, return_type='text', **kwargs).strip()
        return response

    return llm_function