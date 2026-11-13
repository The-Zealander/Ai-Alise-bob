import ollama


def model_llama(msg, mdl):
    """Send a prompt to a local Ollama model and return the model output."""
    messages = [
        {'role': 'system', 'content': 'You are an honest and helpful assistant. Return only code when asked for code.'},
        {'role': 'user', 'content': msg}
    ]

    response = ollama.chat(model=mdl, messages=messages)
    return response.message.content.strip()