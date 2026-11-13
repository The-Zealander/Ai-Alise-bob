from Model_lfm3 import HEAVY_MODEL, run_heavy_model

__all__ = ["HEAVY_MODEL", "run_heavy_model"]

import ollama

def model_llama(msg, mdl):

    msg1 = " Only return code."
    messages = [
        {'role': 'system', 'content': 'You are an honest and helpful assistant.'},
        {'role': 'user', 'content': msg + msg1}
    ]

    response = ollama.chat(model=mdl, messages=messages)
    rsp = f"Assistant: {mdl}" + response.message.content
    
    return rsp
