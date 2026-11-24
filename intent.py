"""
Understands what the user means instead of matching exact phrases.

Each built-in intent gets a score from simple language rules:
    STRONG - clearly this intent, handled right away
    WEAK   - might be this intent, qwen is asked to decide
    0      - not this intent

qwen2.5:0.5b is too small to route every sentence on its own (in
testing it sent "what is the capital of France" to the camera), so it
only breaks ties between the intents the rules already suspect. It can
never pick an intent the rules gave a score of 0.

Nothing in this file touches hardware, so it can be tested anywhere.
"""

import difflib
import re

import ollama

from config import OLLAMA_MODEL, OLLAMA_KEEP_ALIVE, WAKEWORD

STRONG = 2
WEAK = 1

CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "whos": "who is",
    "where's": "where is",
    "wheres": "where is",
    "when's": "when is",
    "how's": "how is",
    "that's": "that is",
    "thats": "that is",
    "it's": "it is",
    "there's": "there is",
    "here's": "here is",
    "what're": "what are",
    "who're": "who are",
    "they're": "they are",
    "you're": "you are",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "i'll": "i will",
    "you'll": "you will",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "let's": "let us",
    "today's": "today",
    "todays": "today",
}


def normalize(text):
    """Lowercase, drop punctuation and expand contractions."""

    text = text.lower().replace("’", "'").replace("`", "'")

    text = re.sub(r"[^a-z0-9' ]+", " ", text)

    words = []

    for word in text.split():
        word = CONTRACTIONS.get(word, word)
        words.append(word.replace("'", ""))

    return " ".join(" ".join(words).split())


def has_phrase(text, phrases):
    """True if any phrase appears in text as whole words."""

    padded = f" {text} "

    return any(f" {phrase} " in padded for phrase in phrases)


def has_unnegated_phrase(text, phrases):
    """Like has_phrase, but ignores "don't go to sleep" style uses."""

    padded = f" {text} "

    for phrase in phrases:

        index = padded.find(f" {phrase} ")

        if index == -1:
            continue

        before = padded[:index].split()[-2:]

        if not ({"not", "never", "dont"} & set(before)):
            return True

    return False


def index_of_any(words, targets, start=0):

    for i in range(start, len(words)):
        if words[i] in targets:
            return i

    return -1


def follows_within(words, firsts, seconds, gap):
    """True if a word from seconds comes up to gap words after one from firsts."""

    for i, word in enumerate(words):

        if word not in firsts:
            continue

        if any(w in seconds for w in words[i + 1:i + 1 + gap]):
            return True

    return False


# ---------------------------------------------------------------------------
# Wake word and mode commands (used by main.py)
# ---------------------------------------------------------------------------

WAKE_PREFIXES = {"hey", "hi", "hello", "ok", "okay", "yo", "um", "uh", "so", "oh"}


def split_wakeword(text):
    """
    Find the wake word at the start of what was said, allowing a
    greeting first ("hey friday, what time is it").

    Returns (found, rest_of_sentence).
    """

    words = text.split()

    for i, word in enumerate(words[:3]):

        if word == WAKEWORD:
            return True, " ".join(words[i + 1:])

        if word not in WAKE_PREFIXES:
            break

    return False, text


FILLER_WORDS = {
    WAKEWORD, "ok", "okay", "well", "alright", "then", "now", "please",
    "sir", "thanks", "thank", "you", "for", "today", "and", "so", "right",
}

GOODBYE_PHRASES = {
    "goodbye", "good bye", "bye", "bye bye", "exit", "quit",
    "shut down", "shutdown", "power off", "turn off", "switch off",
    "see you", "see you later", "see ya", "later", "good night",
    "goodnight", "that is all", "that will be all", "go offline",
}


def is_goodbye(text):
    """
    Shutting down is hard to undo, so this only matches when the
    whole sentence is a goodbye, e.g. "okay bye friday" but not
    "turn off the lights".
    """

    words = normalize(text).split()

    while words and words[0] in FILLER_WORDS - {"you"}:
        words.pop(0)

    while words and words[-1] in FILLER_WORDS - {"see"}:
        words.pop()

    return " ".join(words) in GOODBYE_PHRASES


QUIET_PHRASES = [
    "be quiet", "quiet mode", "stop listening", "stop always listening",
    "go to sleep", "go back to sleep", "sleep mode", "shut up",
    "stop talking", "wake word mode", "only listen when i call you",
    "only when i call you", "only respond to your name", "silence",
    "take a break", "stand by", "standby",
]

ALWAYS_LISTEN_PHRASES = [
    "always listen", "always be listening", "keep listening",
    "listen all the time", "listen continuously", "listen to everything",
    "stay awake", "stay on", "do not sleep", "stop sleeping",
    "conversation mode", "always on", "no wake word", "without the wake word",
    "without your name",
]


def is_quiet_command(text):

    return has_unnegated_phrase(normalize(text), QUIET_PHRASES)


def is_always_listen_command(text):

    return has_unnegated_phrase(normalize(text), ALWAYS_LISTEN_PHRASES)


CANCEL_PHRASES = [
    "cancel", "never mind", "nevermind", "forget it", "forget about it",
    "do not save", "do not bother", "abort", "stop", "no thanks",
]


def is_cancel(text):

    return has_phrase(normalize(text), CANCEL_PHRASES)


NAME_PREFIX = re.compile(
    r"^(?:(?:okay|ok|well|so|um|uh|yes|yeah|sure)[\s,]+)*"
    r"(?:(?:his|her|their|the|my|this person's|this persons)\s+name\s+is"
    r"|(?:it|this|that|he|she)\s*(?:is|'s|s)"
    r"|(?:i\s+am|i'm|im)"
    r"|(?:call\s+(?:him|her|them|me))"
    r"|(?:name\s+is))\s+",
    re.IGNORECASE
)


def extract_person_name(raw_text):
    """
    "His name is John." -> "John"
    "This is Sarah"     -> "Sarah"
    "Mike"              -> "Mike"
    """

    name = raw_text.strip(" .,!?:;")

    name = NAME_PREFIX.sub("", name)

    name = re.sub(
        r"[\s,]+(?:please|sir|friday|thanks|thank you)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name.strip(" .,!?:;")


# ---------------------------------------------------------------------------
# Built-in command intents
# ---------------------------------------------------------------------------

TIME_SUBJECTS = {
    "time": "time", "clock": "time",
    "date": "date",
    "day": "day", "weekday": "day",
    "month": "month",
    "year": "year",
}

# Words that may follow "what time" without changing the meaning,
# e.g. "what time is it right now". Anything else ("what time does
# the shop close", "what day is christmas") is not a clock question.
TIME_FOLLOWERS = {
    "is", "it", "now", "right", "today", "currently", "please", "sir",
    "exactly", "at", "the", "moment", "of", "week", "this", "we", "are",
    "in", "here", "friday", "again",
}

# "what a day", "how much time", "last year" are not clock questions.
TIME_BAD_BEFORE = {
    "a", "much", "more", "less", "some", "spare", "free", "good", "nice",
    "great", "every", "each", "one", "first", "last", "next", "whole",
}

TIME_CUES = {
    "what", "which", "tell", "know", "current", "give", "say", "check",
    "whats", "got",
}

TIME_BLOCKERS = {
    "does", "do", "did", "will", "would", "should", "until", "since",
    "zone", "long", "ago", "times", "birthday", "last", "next", "tomorrow",
    "yesterday", "after", "before", "when", "spend", "free", "busy",
}


def score_time(words):
    """Returns (score, kind) where kind is time/date/day/month/year."""

    text = " ".join(words)

    if text in ("what is today", "what is it today", "what day is it today"):
        return STRONG, "day" if "day" in words else "date"

    i = index_of_any(words, TIME_SUBJECTS)

    if i == -1:
        return 0, None

    kind = TIME_SUBJECTS[words[i]]

    if i > 0 and words[i - 1] in TIME_BAD_BEFORE:
        return 0, None

    # "what's the time in Tokyo" -> leave it to the LLM.
    if "in" in words[i + 1:] and words[-1] not in TIME_FOLLOWERS:
        return 0, None

    after = words[i + 1:]

    if any(w in TIME_BLOCKERS for w in after):
        return 0, None

    has_cue = bool(TIME_CUES & set(words[:i])) or i <= 1

    if not has_cue:
        return 0, None

    if all(w in TIME_FOLLOWERS for w in after):
        return STRONG, kind

    if len(after) <= 3:
        return WEAK, kind

    return 0, None


# "look", "watch" and "view" are left out on purpose: "do you watch
# football" is not a camera question. "what are you looking at" is
# covered by SCENE_PHRASES.
SEE_VERBS = {
    "see", "seeing", "spot", "notice", "observe", "detect", "perceive",
}

SCENE_PHRASES = [
    "in front of you", "around you", "your surroundings", "the surroundings",
    "look around", "near you", "your camera", "your view", "on camera",
    "you looking at", "the scene", "what is around", "what is here",
    "what is there", "what is nearby",
]

SEE_BLOCKERS = {
    "mean", "later", "soon", "tomorrow", "up", "forward", "like", "seem",
    "seems", "movie", "film", "show", "episode", "game", "match",
    "problem", "point", "difference", "why", "future",
}

SEE_WEAK_PHRASES = [
    "what is this", "what is that", "what are these", "what are those",
    "on the table", "on the desk", "on my desk", "in my hand",
    "anything interesting", "what am i holding", "describe the room",
    "describe this", "what is going on",
]


def score_see(words):

    text = " ".join(words)

    if SEE_BLOCKERS & set(words):
        return 0

    if has_phrase(text, SCENE_PHRASES):
        return STRONG

    if SEE_VERBS & set(words) and "you" in words:
        return STRONG

    if "describe" in words and has_phrase(
        text, ["what", "room", "area", "surroundings", "view", "scene"]
    ):
        return STRONG

    if has_phrase(text, SEE_WEAK_PHRASES):
        return WEAK

    return 0


PERSON_WORDS = {
    "person", "people", "persons", "face", "faces", "guy", "guys", "man",
    "men", "woman", "women", "girl", "boy", "kid", "kids", "human",
    "humans", "folks", "friend", "individual", "individuals",
}

DEICTIC_WORDS = {
    "this", "that", "these", "those", "he", "she", "they", "him", "her",
    "them", "there", "here", "me", "i",
}

# "who invented the telephone" is a knowledge question, not "who is
# standing in front of you".
WHO_BLOCKERS = {
    "invented", "wrote", "discovered", "founded", "created", "made", "built",
    "painted", "won", "said", "sang", "sings", "directed", "owns",
    "president", "king", "queen", "ceo", "richest", "best", "first",
    "famous", "called", "named", "played", "plays", "was", "were",
    "killed", "designed", "leads", "runs",
}

WHO_PRESENCE_PHRASES = [
    "in front of you", "you see", "you looking at", "standing", "sitting",
    "around you", "in the room", "with me", "next to me", "beside me",
    "behind me", "on camera", "at the door", "over there",
]

RECOGNIZE_VERBS = {"recognize", "recognise", "identify"}

ANYONE_WORDS = {"anyone", "anybody", "someone", "somebody"}


def score_who(words):

    text = " ".join(words)

    if WHO_BLOCKERS & set(words):
        return 0

    # "remember who this is" is face registration.
    if LEARN_VERBS & set(words) and not has_phrase(text, ["do you", "did you"]):
        return 0

    if "who" in words:

        if has_phrase(text, WHO_PRESENCE_PHRASES):
            return STRONG

        i = words.index("who")

        # "who is this", "who are they", "who am i", "who is this guy".
        # Longer ones like "who is she married to" are about someone
        # mentioned in conversation, so they go to the LLM.
        if any(w in DEICTIC_WORDS for w in words[i + 1:i + 4]):

            if len(words) - i <= 4:
                return STRONG

    targets = PERSON_WORDS | DEICTIC_WORDS | ANYONE_WORDS | {"who"}

    if follows_within(words, RECOGNIZE_VERBS, targets, 4):
        return STRONG

    # "do you know who this is", "do you know me"
    know_targets = PERSON_WORDS | {
        "this", "that", "these", "those", "him", "her", "them", "me", "who",
    }

    if follows_within(words, {"know"}, know_targets, 3) and "you" in words:
        return STRONG

    # "do you remember me", "do you remember this person"
    if has_phrase(text, ["do you remember", "you remember"]) and (
        follows_within(words, {"remember"}, PERSON_WORDS | {"me", "him", "her", "them"}, 2)
    ):
        return STRONG

    # "is anyone there", "is somebody here"
    if follows_within(words, {"is", "anyone", "anybody", "someone", "somebody"},
                      {"there", "here", "around"}, 2) and (
            ANYONE_WORDS & set(words)):
        return STRONG

    return 0


PRESENCE_WORDS = {
    "here", "there", "see", "visible", "present", "front", "room", "around",
    "detect", "camera", "currently", "now", "count",
}

COUNT_BLOCKERS = {
    "live", "lived", "died", "die", "work", "world", "country", "city",
    "earth", "planet", "history", "attend", "speak", "use", "visit",
    "year", "years", "watched", "own",
}


def score_count(words):

    text = " ".join(words)

    if COUNT_BLOCKERS & set(words):
        return 0

    if has_phrase(text, ["how many"]):

        if not (PERSON_WORDS & set(words) or has_phrase(text, ["of us", "of you"])):
            return 0

        if PRESENCE_WORDS & set(words) or has_phrase(text, ["are there"]):
            return STRONG

        if len(words) <= 6:
            return WEAK

        return 0

    if follows_within(words, {"count"}, PERSON_WORDS, 2):
        return STRONG

    return 0


# "remember him", "learn my face", "scan me"
LEARN_VERBS = {
    "remember", "learn", "memorize", "memorise", "register", "scan",
    "enroll", "enrol",
}

# These only count with a face or person word: "save me a seat" and
# "add this to my list" are not face registration.
SAVE_VERBS = {"save", "store", "add", "record"}

LEARN_PRONOUNS = {"him", "her", "them", "me"}


def score_learn_face(words):

    text = " ".join(words)

    # Questions like "do you remember me" are recognition, not learning.
    if has_phrase(text, ["do you", "did you", "can you tell", "have you"]) and (
            "remember" in words or "know" in words):
        return 0

    # "remember that my car is red" is a fact, not a face.
    if "face" not in words and "faces" not in words and "who" not in words and (
            {"is", "are", "was", "were"} & set(words)):
        return 0

    any_verb = LEARN_VERBS | SAVE_VERBS

    if follows_within(words, any_verb, {"face", "faces"}, 4):
        return STRONG

    if follows_within(words, any_verb, PERSON_WORDS, 3):
        return STRONG

    if follows_within(words, LEARN_VERBS, LEARN_PRONOUNS, 2):
        return STRONG

    # "i want you to meet my friend", "meet this guy"
    if follows_within(words, {"meet"}, {"my", "this", "him", "her"}, 2):
        return STRONG

    # "remember who this is", "remember this": probably, but let qwen check.
    if follows_within(words, LEARN_VERBS, {"this", "who"}, 2):
        return WEAK

    return 0


def score_intents(text):
    """
    Score every built-in intent.

    Returns {intent: score} for intents with score > 0, and the time
    kind (time/date/day/month/year) if time scored.
    """

    words = normalize(text).split()

    if not words:
        return {}, None

    scores = {}

    time_score, time_kind = score_time(words)

    for name, score in (
        ("time", time_score),
        ("learn_face", score_learn_face(words)),
        ("count", score_count(words)),
        ("who", score_who(words)),
        ("see", score_see(words)),
    ):
        if score:
            scores[name] = score

    return scores, time_kind


# Most specific first: "how many people do you see" is a count, not a
# scene description, even though both match.
INTENT_PRIORITY = ["learn_face", "count", "who", "time", "see"]

CLASSIFIER_SYSTEM = (
    "Classify the user's sentence into exactly one label. Labels: "
    "time, see, who, count, learn_face, chat. Answer with the label only."
)

CLASSIFIER_EXAMPLES = [
    ("what time is it", "time"),
    ("what day is it today", "time"),
    ("what do you see", "see"),
    ("what is on my desk", "see"),
    ("who is this", "who"),
    ("do you recognize him", "who"),
    ("how many people are in the room", "count"),
    ("remember this face", "learn_face"),
    ("tell me a story", "chat"),
    ("what is the biggest ocean", "chat"),
    ("who is she married to", "chat"),
    ("what day is christmas", "chat"),
    ("hi there", "chat"),
]


def ask_qwen_for_intent(text, candidates):
    """
    Let qwen pick between the suspected intents and plain chat.
    Returns an intent from candidates, or None for chat.
    """

    messages = [{"role": "system", "content": CLASSIFIER_SYSTEM}]

    for example, label in CLASSIFIER_EXAMPLES:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": label})

    messages.append({"role": "user", "content": text})

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0, "num_predict": 5},
            keep_alive=OLLAMA_KEEP_ALIVE
        )

    except Exception as error:
        print(f"[FRIDAY] Intent check failed: {error}")
        return None

    answer = response["message"]["content"].strip().lower()

    match = re.match(r"[a-z_]+", answer)

    label = match.group(0) if match else ""

    print(f"[FRIDAY] qwen thinks {text!r} is: {label or answer!r}")

    return label if label in candidates else None


def detect_intent(text, use_llm=True):
    """
    Work out which built-in command the user means.

    Returns (intent, time_kind). intent is None when the sentence
    should go to the LLM as normal conversation.
    """

    scores, time_kind = score_intents(text)

    if not scores:
        return None, None

    strong = [name for name in INTENT_PRIORITY if scores.get(name) == STRONG]

    if strong:
        return strong[0], time_kind

    weak = [name for name in INTENT_PRIORITY if scores.get(name) == WEAK]

    if not use_llm:
        return None, None

    intent = ask_qwen_for_intent(normalize(text), weak)

    return intent, time_kind


# ---------------------------------------------------------------------------
# Fuzzy word matching (used by memory.py)
# ---------------------------------------------------------------------------

def _stem(word):

    if len(word) > 3 and word.endswith("s"):
        return word[:-1]

    return word


def words_match(a, b):
    """'color' ~ 'colour', 'keys' ~ 'key', 'favorite' ~ 'favourite'."""

    if a == b or _stem(a) == _stem(b):
        return True

    if len(a) >= 4 and len(b) >= 4:
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

    return False
