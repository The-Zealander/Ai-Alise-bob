import ollama
import model_llama

LFM3_MODEL = 'lfm3'
DELEGATE_PREFIX = 'llama3.2'

SYSTEM_PROMPT = (
    "You are a helpful assistant and are a part of a small agentic program "
    "consisting of you and llama3.2, a larger model. You are responsible for "
    "delegating tasks between yourself and llama3.2. Only delegate a task if "
    "you think it is too complex for you to reliably answer. To delegate, begin "
    f"your response with '{DELEGATE_PREFIX}', and make the rest of your response "
    "the prompt for llama3.2."
)

FOLLOW_UP = (
    "llama3.2 replied:\n{reply}\n\n"
    "Forward this to the user if it is good enough (reply normally), or start "
    f"your message with '{DELEGATE_PREFIX}' to ask llama3.2 again."
)


def new_conversation():
    return [{'role': 'system', 'content': SYSTEM_PROMPT}]


def call_lfm3(messages):
    """Send the conversation to lfm3, record its reply in `messages`, return it."""
    if messages is None:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    response = ollama.chat(model=LFM3_MODEL, messages=messages)
    answer = response.message.content.strip()
    answer[len(DELEGATE_PREFIX):].lstrip(' :,-\n')
    messages.append({'role': 'assistant', 'content': answer})
    
    return answer


def main():
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    print("Type /bye or quit to exit.")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in {'/bye', '/quit', 'quit'}:
            print("Goodbye!")
            break
        if not prompt:
            continue

    answer = call_lfm3(messages)

    for _ in range(5):
        if answer.strip().lower().startswith(DELEGATE_PREFIX):
            prompt = answer.strip()[len(DELEGATE_PREFIX):].lstrip(' :,-\n')
            llama_response = model_llama.call_llama(prompt)
            messages.append({
                'role': 'user',
                'content': FOLLOW_UP.format(reply=llama_response),
            })

            answer = call_lfm3(messages)

        else: break

        print(answer)

if __name__ == '__main__':
    main()
