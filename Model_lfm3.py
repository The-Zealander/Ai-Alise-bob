import ollama


SYSTEM_PROMPT = (
    "You are named 'Alise'. You are the interaction layer of a Python project. "
    "You are responsible for delegating tasks between yourself and another model named 'llama3.2' "
    "if the tasks are heavy, or yourself if it is an easy task. If llama does a task, you are then "
    "returned the product and are to describe it for the user and give it to the user."
)


def call_lfm3(prompt, messages=None):
    """Call lfm3, optionally continuing an existing conversation."""
    if messages is None:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    messages.append({'role': 'user', 'content': prompt})
    response = ollama.chat(model='lfm3', messages=messages)
    answer = response.message.content.strip()
    messages.append({'role': 'assistant', 'content': answer})
    return answer


def main():
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    print("Chat with Alise. Type /bye or quit to exit.")

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

        result = call_lfm3(prompt, messages)
        print(f"Assistant: lfm3\n{result}")


if __name__ == '__main__':
    main()
