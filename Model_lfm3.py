import ollama
import Model_llama


SYSTEM_PROMPT = (
    """You are a helpful assistant and are apart of a small agentic program consisting of you and llama3.2.
    You are responsible for delegating tasks between yourself and the llama3.2 model.
    You should only delegate
    a task to llama3.2 if you think the given task is too complex
    for you to realiably answer. With this in mind, if you want
    to delegate a task to llama3.2, please begin your response
    with 'llama3.2', and then the rest of your response should
    be a prompt for the llama3.2 model.
    """
)


def call_lfm3(prompt, messages=None):
    """Call lfm3, optionally continuing an existing conversation."""
    if messages is None:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    messages.append({'role': 'user', 'content': prompt})
    response = ollama.chat(model='lfm3', messages=messages)
    answer = response.message.content.strip()
    messages.append({'role': 'assistant', 'content': answer})
    print(f"LFM3 answer: {answer} \n\n")
    while (answer[:8] == 'llama3.2'):
        answer = Model_llama.ai_model(answer, 'llama3.2')
        print(f"llama3.2 answer: {answer} \n\n")
        answer = Model_llama.ai_model(answer, 'lfm3').partition("(*&*)")[0]
        print(f"LFM3 answer: {answer} \n\n")

    # if (answer[:8] == 'llama3.2'):
    #     answer = Model_llama.ai_model(answer, 'llama3.2')

    return answer


def main():
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    print("Chat with Alice. Type /bye or quit to exit.")

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
        print(result)


if __name__ == '__main__':
    main()
