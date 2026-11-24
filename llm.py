import re

from collections import deque

import ollama

from config import CALLNAME, OLLAMA_MODEL, OLLAMA_KEEP_ALIVE
from memory import memory_summary

# The last few exchanges, so follow-up questions ("how old is he?")
# have something to refer to. Kept short: qwen2.5:0.5b is small and
# runs on the Jetson.
HISTORY_TURNS = 3

history = deque(maxlen=HISTORY_TURNS)


def add_to_history(user_text, reply):
    """Record an exchange, including ones answered by built-in commands."""

    history.append((user_text, reply))


def clear_history():

    history.clear()

# Short example exchanges that set FRIDAY's tone. They're sent as
# real chat turns, which qwen follows far better than a transcript
# pasted into one message.
examples = [
    ("Hello", f"Hello, {CALLNAME}."),
    ("Who are you", f"I am Friday, your personal assistant, {CALLNAME}."),
    ("what is your purpose", f"To assist you efficiently, {CALLNAME}."),
    ("are you here", f"Yes, I am here, {CALLNAME}."),
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

    # Lowercase letter or digit before the full stop, so initials
    # like "J.K. Rowling" aren't treated as the end of a sentence.
    sentences = re.split(
        r'(?<=[a-z0-9)"\'][.!?]) +',
        text.strip()
    )

    # num_predict can cut the last sentence off mid-word; drop it if
    # there's a complete one before it.
    if len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?")):
        sentences = sentences[:-1]

    return " ".join(sentences[:max_sentences])


def build_messages(prompt):

    system = (
        "You are FRIDAY, a personal voice assistant. "
        f"Always address the user as '{CALLNAME}'. "
        "Answer in one or two short sentences. Be concise and direct. "
        "Use the earlier conversation to understand follow-up questions."
    )

    facts = memory_summary()

    if facts:
        system += (
            " Things the user told you to remember: "
            + "; ".join(facts)
            + "."
        )

    messages = [{"role": "system", "content": system}]

    for user_text, reply in list(examples) + list(history):
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": reply})

    messages.append({"role": "user", "content": prompt})

    return messages


def ask_ai(prompt):

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=build_messages(prompt),
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
