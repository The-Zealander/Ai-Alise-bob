from Model_lfm3 import HEAVY_MODEL, run_heavy_model

__all__ = ["HEAVY_MODEL", "run_heavy_model"]

import ollama

msg = input()
msg1 = " Only return code."
messages = [
    {'role': 'system', 'content': 'You are an honest and helpful assistant.'},
    {'role': 'user', 'content': msg + msg1}
]

mdl = 'llama3.2'
response = ollama.chat(model=mdl, messages=messages)
print(f"Assistant: {mdl}")
print(response.message.content)
