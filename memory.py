import json
import re

from config import CALLNAME, MEMORY_FILE
from intent import normalize, words_match


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


# "remember that ...", "please memorize ...", "make a note that ..."
REMEMBER_PREFIX = (
    r"^(?:(?:can you|could you|please|i want you to|you should|"
    r"i need you to|hey|okay|ok)\s+)*"
    r"(?:remember|memorize|memorise|note|keep in mind|do not forget|"
    r"never forget|make a note|take a note|save|store|write down)"
    r"(?:\s+(?:that|this|of that|for me that))?\s+"
)

# "... remember that", "... do not forget it"
REMEMBER_SUFFIX = (
    r"\s+(?:(?:please|okay|so)\s+)?"
    r"(?:remember that|remember this|remember it|do not forget that|"
    r"do not forget it|do not forget|keep that in mind|keep it in mind|"
    r"note that|save that|make a note of that|make a note)"
    r"(?:\s+please)?$"
)

FACT_BODY = r"(.+?)\s+(?:is|are|was|were)\s+(.+)"

# Words that make a sentence a question, not a fact to store.
QUESTION_STARTS = {"what", "where", "who", "when", "which", "how", "why", "do", "did"}


def parse_fact(text):
    """
    Pull (key, value) out of a sentence like:
        "remember my favorite color is red"
        "remember that the wifi password is banana"
        "please memorize that my car is parked on level 2"
        "my keys are in the drawer, remember that"
    Returns None if the sentence isn't asking to store a fact.
    """

    text = normalize(text)

    words = text.split()

    if not words or words[0] in QUESTION_STARTS:
        return None

    match = re.match(REMEMBER_PREFIX + FACT_BODY + "$", text)

    if match is None:
        match = re.match("^" + FACT_BODY + REMEMBER_SUFFIX, text)

    if match is None:
        return None

    key = match.group(1).strip()
    value = match.group(2).strip()

    value = re.sub(r"\s+(?:please|okay|sir)$", "", value).strip()

    if not key or not value:
        return None

    return key, value


PERSPECTIVE = {
    "my": "your",
    "mine": "yours",
    "me": "you",
    "i": "you",
    "myself": "yourself",
    "our": "your",
    "your": "my",
}


def from_your_side(phrase):
    """'my wifi password' -> 'your wifi password'."""

    return " ".join(PERSPECTIVE.get(word, word) for word in phrase.split())


def describe_fact(key, value):

    verb = "are" if key.split()[-1].endswith("s") and not key.endswith("ss") else "is"

    return f"{from_your_side(key)} {verb} {value}"


def remember_fact(text):

    fact = parse_fact(text)

    if fact is None:
        return None

    key, value = fact

    memory = load_memory()

    memory[key] = value

    save_memory(memory)

    return f"Okay {CALLNAME}, I'll remember that {describe_fact(key, value)}."


# Words ignored when matching a question to a saved key.
STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "was", "were", "to", "for", "in",
    "on", "at", "and", "what", "where", "who", "when", "which", "how", "do",
    "did", "you", "tell", "me", "remember", "recall", "about", "know",
    "please", "again", "it", "that", "this", "remind", "said", "told",
    "i", "say", "can", "could", "would",
}

OWNER_WORDS = {"my", "your", "his", "her", "our", "their"}

# Questions that are clearly about something the user told FRIDAY
# earlier, so "I don't know" is the right answer if nothing matches.
EXPLICIT_RECALL = [
    r"^recall\b",
    r"^remind me\b",
    r"\bwhat did i (?:tell|say)\b",
    r"\bdo you remember\b",
    r"\bwhat do you remember\b",
    r"\bdid i (?:tell|say)\b",
]

QUESTION_WORDS = {"what", "where", "who", "when", "which", "how", "tell", "recall", "remind", "do"}


def find_memory_key(question, memory):
    """
    Return the saved key the question is about, or None.

    'what was my wifi password' matches 'my wifi password' and
    'what is my favourite colour' matches 'my favorite color'.
    """

    words = question.split()

    best_key = None
    best_size = 0

    for key in memory:

        key_words = normalize(key).split()

        # "my name" must not answer "what is your name".
        owners = OWNER_WORDS & set(key_words)

        if owners and not owners <= set(words):
            continue

        content = [w for w in key_words if w not in STOPWORDS | OWNER_WORDS]

        if not content:
            continue

        if all(any(words_match(c, w) for w in words) for c in content):

            if len(content) > best_size:
                best_key = key
                best_size = len(content)

    return best_key


def recall_fact(text):

    question = normalize(text)

    words = question.split()

    if not words:
        return None

    explicit = any(re.search(pattern, question) for pattern in EXPLICIT_RECALL)

    is_question = explicit or words[0] in QUESTION_WORDS

    if not is_question:
        return None

    memory = load_memory()

    key = find_memory_key(question, memory)

    if key is not None:

        fact = describe_fact(key, memory[key])

        return f"{fact[0].upper()}{fact[1:]}, {CALLNAME}."

    # "do you remember my birthday" with nothing saved: say so,
    # rather than letting the LLM make something up. Questions about
    # people ("do you remember me") are handled by face recognition.
    if explicit and not re.search(r"\b(?:me|him|her|them|person|face|guy|man|woman)$", question):
        return f"I don't have that information, {CALLNAME}."

    return None


def memory_summary(limit=10):
    """Saved facts as short sentences, for the LLM's context."""

    memory = load_memory()

    return [
        describe_fact(key, value)
        for key, value in list(memory.items())[-limit:]
    ]
