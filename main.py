import sys
import ollama
from pydantic import BaseModel, Field

INTERPRETER_MODEL = "lfm3"
HEAVY_MODEL = "qwen3.5:9b"

INTERPRETER_SYSTEM_PROMPT = """
You are the front-end interpreter and prompt router for a two-model local AI assistant.
Analyze the user's latest message alongside the conversation history.

Decide whether the task requires the heavy reasoning/coding model (qwen3.5:9b):
- Set needs_heavy_model = false for: greetings, simple factual lookups, brief summaries,
  clarifying questions, or basic single-step requests you can answer accurately yourself.
  Put your full response in `direct_reply` and leave `improved_prompt` empty.
- Set needs_heavy_model = true for: writing or debugging code, software architecture,
  multi-step logic/math, deep analysis, or complex project planning.
  Leave `direct_reply` empty, and write a comprehensive, unambiguous, well-structured prompt
  in `improved_prompt` that preserves all user requirements, adds helpful technical constraints,
  and incorporates any relevant context from earlier in the conversation.
""".strip()

HEAVY_SYSTEM_PROMPT = """
You are a senior software engineer and analytical problem solver.
Follow the detailed specification provided in the user message thoroughly, accurately, and concisely.
""".strip()


class RouteDecision(BaseModel):
    needs_heavy_model: bool = Field(
        description="True if the task needs complex coding, math, architecture, or multi-step reasoning."
    )
    reasoning: str = Field(
        description="One-sentence explanation of why this task is handled directly or escalated."
    )
    direct_reply: str = Field(
        default="",
        description="Full answer to the user if needs_heavy_model is False; otherwise empty."
    )
    improved_prompt: str = Field(
        default="",
        description="Refined, detailed instructions for qwen3.5:9b if needs_heavy_model is True; otherwise empty."
    )


def interpret_and_route(chat_history: list[dict[str, str]], user_input: str) -> RouteDecision:
    """Use lfm3 with structured outputs to triage and optionally rewrite the prompt."""
    messages = [
        {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
        *chat_history[-6:],  # Keep recent context lightweight for fast triage
        {"role": "user", "content": user_input},
    ]

    response = ollama.chat(
        model=INTERPRETER_MODEL,
        messages=messages,
        format=RouteDecision.model_json_schema(),
        options={"temperature": 0.0},  # Deterministic routing & JSON generation
    )

    return RouteDecision.model_validate_json(response.message.content)


def run_heavy_model(chat_history: list[dict[str, str]], improved_prompt: str) -> str:
    """Stream the response from qwen3.5:9b using the upgraded instructions."""
    messages = [
        {"role": "system", "content": HEAVY_SYSTEM_PROMPT},
        *chat_history,
        {"role": "user", "content": improved_prompt},
    ]

    stream = ollama.chat(
        model=HEAVY_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3},
    )

    full_response: list[str] = []
    for chunk in stream:
        token = chunk.message.content
        print(token, end="", flush=True)
        full_response.append(token)
    print()  # Newline after stream finishes

    return "".join(full_response)


def main() -> None:
    chat_history: list[dict[str, str]] = []
    print(f"Interactive Router Ready | Interpreter: {INTERPRETER_MODEL} -> Heavy: {HEAVY_MODEL}")
    print("Type 'exit' or 'quit' to stop, or '/clear' to reset memory.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "/clear":
            chat_history.clear()
            print("[Memory cleared]\n")
            continue

        try:
            print(f"\n[{INTERPRETER_MODEL} analyzing request...]")
            decision = interpret_and_route(chat_history, user_input)
            print(f"[Triage]: {decision.reasoning}")

            if not decision.needs_heavy_model:
                reply = decision.direct_reply.strip()
                print(f"\nAssistant ({INTERPRETER_MODEL}):\n{reply}\n")
            else:
                escalated_prompt = decision.improved_prompt.strip() or user_input
                print(f"\n[Escalating to {HEAVY_MODEL} with improved instructions:]")
                print(f"--- Prompt Sent ---\n{escalated_prompt}\n-------------------\n")
                print(f"Assistant ({HEAVY_MODEL}):")
                reply = run_heavy_model(chat_history, escalated_prompt)
                print()

            # Save original user turn and final assistant output for multi-turn continuity
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": reply})

        except ollama.ResponseError as err:
            print(f"\n[Ollama Error]: {err.error}", file=sys.stderr)
            print(f"Verify both models are pulled: `ollama pull {INTERPRETER_MODEL}` and `ollama pull {HEAVY_MODEL}`\n")
        except Exception as exc:
            print(f"\n[Connection Error]: Could not reach Ollama at localhost:11434 ({exc})\n", file=sys.stderr)


if __name__ == "__main__":
    main()