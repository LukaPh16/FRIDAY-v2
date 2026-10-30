import json
import re

from config import CALLNAME, MEMORY_FILE


def load_memory():

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (OSError, json.JSONDecodeError):

        return {}


def save_memory(memory):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            memory,
            f,
            indent=4
        )


def remember_fact(text):

    memory = load_memory()

    text = text.lower().strip()

    match = re.search(
        r"remember[,]?\s+(.+?)\s+is\s+(.+?)[.!?]?$",
        text
    )

    if match:

        key = match.group(1).strip()

        value = match.group(2).strip()

        memory[key] = value

        save_memory(memory)

        return f"Remembered, {CALLNAME}."

    return None


def recall_fact(text):

    memory = load_memory()

    text = text.lower().strip()

    match = re.search(
        r"recall\s+(.+?)[.!?]?$",
        text
    )

    if match:

        key = match.group(1).strip()

        if key in memory:

            return f"{memory[key]}, {CALLNAME}."

        return f"I don't have that information, {CALLNAME}."

    return None
