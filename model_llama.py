import ollama

LLAMA_MODEL = 'llama3.2'

LLAMA_SYSTEM = (
    "You are a helpful assistant and are a part of a small agentic program "
    "consisting of you and a smaller model, lfm3. lfm3 decides whether a task "
    "is too complex for itself, and only then sends it to you. Answer the "
    "prompt you receive as completely and accurately as you can."
)


def call_llama(prompt):
    
    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[
            {'role': 'system', 'content': LLAMA_SYSTEM},
            {'role': 'user', 'content': prompt},
        ],
    )

    return response.message.content.strip()
