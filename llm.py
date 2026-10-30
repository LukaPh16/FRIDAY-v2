import re

import ollama

from config import CALLNAME, OLLAMA_MODEL, OLLAMA_KEEP_ALIVE

examples = [
    "User: Hello\nAI: Hello, sir.",
    "User: Who are you\nAI: I am Friday.",
    "User: who are you\nAI: I am your personal assistant, sir.",
    "User: what is your purpose\nAI: To assist you efficiently, sir.",
    "User: are you here\nAI: Yes, I am here, sir."
]


def warmup_ollama():

    print("[FRIDAY] Warming up Ollama...")

    ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": "hi"
            }
        ],
        options={
            "num_predict": 1
        },
        keep_alive=OLLAMA_KEEP_ALIVE
    )

    print("[FRIDAY] Ollama ready!")


def enforce_callname(text):

    if CALLNAME.lower() not in text.lower():

        text = text.strip()

        if text.endswith((".", "!", "?")):

            text = text[:-1] + f", {CALLNAME}."

        else:

            text += f", {CALLNAME}"

    return text


def limit_sentences(text, max_sentences=2):

    sentences = re.split(
        r'(?<=[.!?]) +',
        text
    )

    return " ".join(sentences[:max_sentences])


def ask_ai(prompt):

    example_text = "\n".join(
        examples[-10:]
    )

    full_prompt = (
        f"You must ALWAYS address the user as '{CALLNAME}'. "
        f"Every answer MUST include '{CALLNAME}'. "
        "You must answer in MAXIMUM 3 short sentences. "
        "Be concise and direct.\n\n"
        + example_text
        + f"\nUser: {prompt}\nAI:"
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        options={
            "num_predict": 30,
            "temperature": 0.3
        },
        keep_alive=OLLAMA_KEEP_ALIVE
    )

    reply = response["message"]["content"]

    reply = limit_sentences(reply)

    reply = enforce_callname(reply)

    return reply
