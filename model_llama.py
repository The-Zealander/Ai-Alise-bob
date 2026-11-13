import ollama

def model_llama(msg, mdl):

    msg1 = " Only return code."
    messages = [
        {'role': 'system', 'content': 'You are an honest and helpful assistant.'},
        {'role': 'user', 'content': msg + msg1}
    ]

    response = ollama.chat(model=mdl, messages=messages)
    rsp = f"Assistant: {mdl} \n" + response.message.content
    
    return rsp
