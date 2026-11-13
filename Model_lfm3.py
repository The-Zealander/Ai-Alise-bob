import base64
import subprocess
import sys
from typing import Callable

import ollama
from pydantic import BaseModel, Field

INTERPRETER_MODEL = "lfm3"
HEAVY_MODEL = "llama3.2"

INTERPRETER_SYSTEM_PROMPT = """
You are the conversational front end for a local AI assistant. Analyze the latest
user message with the conversation history, then choose the best way to answer.

Reply in the user's language and sound natural, clear, and friendly.
- For greetings, simple questions, brief summaries, and straightforward requests,
  answer the user fully in `direct_reply`. Do not mention routing or model names.
- For coding, deep analysis, multi-step reasoning, or other demanding tasks, set
  `needs_heavy_model` to true and rewrite the request in `improved_prompt`.
  Preserve the user's requirements and useful conversation context. Leave
  `direct_reply` empty.

Keep `reasoning` to one short internal sentence. It is not shown to the user.
""".strip()

HEAVY_SYSTEM_PROMPT = """
You are a helpful, capable assistant responding to a request that needs deeper
reasoning. Follow the user's requirements carefully, respond in their language,
and keep the answer natural, clear, and useful.
""".strip()


class RouteDecision(BaseModel):
    needs_heavy_model: bool = Field(
        description="True if the task needs complex coding, analysis, or multi-step reasoning."
    )
    reasoning: str = Field(
        description="One-sentence internal explanation of the routing decision."
    )
    direct_reply: str = Field(
        description="A complete, natural reply to the user when needs_heavy_model is false."
    )
    improved_prompt: str = Field(
        description="A detailed request for llama3.2 when needs_heavy_model is true."
    )


class SpeechUnavailableError(RuntimeError):
    """Raised when Windows speech input or output cannot be used."""


def _run_powershell(script: str, timeout: int) -> str:
    if sys.platform != "win32":
        raise SpeechUnavailableError("Speech input and output are currently supported on Windows only.")

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SpeechUnavailableError("Windows PowerShell was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SpeechUnavailableError("Speech recognition timed out. Please try again.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SpeechUnavailableError(f"Windows speech support failed: {details}") from exc

    return result.stdout.strip()


def listen_for_speech() -> str:
    """Capture one utterance using the local Windows dictation recognizer."""
    script = r"""
Add-Type -AssemblyName System.Speech
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$engine.LoadGrammar([System.Speech.Recognition.DictationGrammar]::new())
$engine.SetInputToDefaultAudioDevice()
$result = $engine.Recognize([TimeSpan]::FromSeconds(15))
if ($null -ne $result) { [Console]::Out.Write($result.Text) }
$engine.Dispose()
"""
    return _run_powershell(script, timeout=25)


def speak_text(text: str) -> None:
    """Speak text through the installed local Windows voice."""
    encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
    script = f"""
Add-Type -AssemblyName System.Speech
$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}'))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Speak($text)
$synth.Dispose()
"""
    _run_powershell(script, timeout=180)


def interpret_and_route(
    chat_history: list[dict[str, str]], user_input: str
) -> RouteDecision:
    """Ask LFM3 for a natural answer or a refined request for the heavy model."""
    messages = [
        {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
        *chat_history[-10:],
        {"role": "user", "content": user_input},
    ]
    response = ollama.chat(
        model=INTERPRETER_MODEL,
        messages=messages,
        format=RouteDecision.model_json_schema(),
        options={"temperature": 0.2},
    )
    decision = RouteDecision.model_validate_json(response.message.content)
    if decision.needs_heavy_model and not decision.improved_prompt.strip():
        decision.improved_prompt = user_input
    if not decision.needs_heavy_model and not decision.direct_reply.strip():
        raise ValueError("LFM3 returned an empty reply. Please try rephrasing your message.")
    return decision


def run_heavy_model(
    chat_history: list[dict[str, str]],
    improved_prompt: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Stream a complex request through Llama 3.2 and return its complete reply."""
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
    response_parts: list[str] = []
    for chunk in stream:
        token = chunk.message.content
        response_parts.append(token)
        if on_token is not None:
            on_token(token)
    return "".join(response_parts)


def _handle_turn(
    chat_history: list[dict[str, str]], user_input: str, voice_output: bool
) -> None:
    decision = interpret_and_route(chat_history, user_input)
    if decision.needs_heavy_model:
        print(f"Assistant ({HEAVY_MODEL}):")
        reply = run_heavy_model(
            chat_history,
            decision.improved_prompt,
            on_token=lambda token: print(token, end="", flush=True),
        )
        print()
    else:
        reply = decision.direct_reply.strip()
        print(f"Assistant ({INTERPRETER_MODEL}):\n{reply}")

    chat_history.extend(
        ({"role": "user", "content": user_input}, {"role": "assistant", "content": reply})
    )
    if voice_output and reply:
        speak_text(reply)


def main() -> None:
    chat_history: list[dict[str, str]] = []
    voice_output = False
    print(f"Alise is ready | {INTERPRETER_MODEL} -> {HEAVY_MODEL}")
    print("Type a message, or use /listen to speak. Commands: /voice on|off, /clear, /quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return

        if not user_input:
            continue
        command = user_input.lower()
        if command in {"/quit", "exit", "quit"}:
            print("Goodbye.")
            return
        if command == "/clear":
            chat_history.clear()
            print("[Conversation cleared]\n")
            continue
        if command in {"/voice on", "/voice off"}:
            voice_output = command.endswith("on")
            print(f"[Spoken replies {'on' if voice_output else 'off'}]\n")
            continue
        if command == "/listen":
            try:
                print("[Listening... speak now]")
                user_input = listen_for_speech()
                if not user_input:
                    print("[No speech was recognized]\n")
                    continue
                print(f"You (voice): {user_input}")
            except SpeechUnavailableError as exc:
                print(f"[Speech input unavailable: {exc}]\n", file=sys.stderr)
                continue

        try:
            _handle_turn(chat_history, user_input, voice_output)
            print()
        except ollama.ResponseError as exc:
            print(f"\n[Ollama error: {exc.error}]", file=sys.stderr)
            print(
                f"Check that Ollama is running and both models are available: "
                f"`ollama pull {INTERPRETER_MODEL}` and `ollama pull {HEAVY_MODEL}`\n",
                file=sys.stderr,
            )
        except SpeechUnavailableError as exc:
            print(f"\n[Speech output unavailable: {exc}]\n", file=sys.stderr)
        except Exception as exc:
            print(f"\n[Could not complete the request: {exc}]\n", file=sys.stderr)


if __name__ == "__main__":
    main()
