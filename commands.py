import datetime

from camera import get_face_count, get_face_results
from config import CALLNAME
from intent import detect_intent
from memory import remember_fact, recall_fact
from objects import describe_scene


TIME_FORMATS = {
    "time": f"The time is %H:%M, {CALLNAME}.",
    "date": f"Today is %B %d, {CALLNAME}.",
    "day": f"Today is %A, {CALLNAME}.",
    "month": f"It is %B, {CALLNAME}.",
    "year": f"It is %Y, {CALLNAME}.",
}


def tell_time(kind):

    now = datetime.datetime.now()

    return now.strftime(TIME_FORMATS.get(kind, TIME_FORMATS["time"]))


def describe_face_count():

    count = get_face_count()

    if count == 1:

        return (
            f"I currently detect one face, "
            f"{CALLNAME}."
        )

    return (
        f"I currently detect {count} faces, "
        f"{CALLNAME}."
    )


def recognize_current_people():

    results = get_face_results()

    if not results:
        return (
            f"I don't currently see anyone's face, "
            f"{CALLNAME}."
        )

    total = len(results)

    known_people = [
        result["name"]
        for result in results
        if result["recognized"]
    ]

    unknown_count = sum(
        1
        for result in results
        if not result["recognized"]
    )

    parts = [
        f"I can see {total} "
        f"{'person' if total == 1 else 'people'}"
    ]

    if known_people:

        # Count how many times each name appears.
        name_counts = {}

        for name in known_people:
            name_counts[name] = name_counts.get(name, 0) + 1

        descriptions = []

        for name, count in name_counts.items():

            if count == 1:
                descriptions.append(name)
            else:
                descriptions.append(
                    f"{name}, appearing {count} times"
                )

        if len(descriptions) == 1:
            known_description = descriptions[0]
        else:
            known_description = (
                ", ".join(descriptions[:-1])
                + " and "
                + descriptions[-1]
            )

        parts.append(
            f"I recognize {known_description}"
        )

    if not known_people:
        parts.append(
            "I don't recognize anyone"
        )

    elif unknown_count == 1:
        parts.append(
            "I don't recognize one person"
        )

    elif unknown_count > 1:
        parts.append(
            f"I don't recognize {unknown_count} people"
        )

    reply = ". ".join(parts) + f", {CALLNAME}."

    return reply


def route_command(user_input):
    """
    Work out what the user means and run the matching built-in command.

    The wording doesn't have to be exact: "what's the time",
    "got the time?" and "could you tell me what time it is" all work.
    See intent.py for how this is decided.

    Returns:
        (reply, wants_face_name)

    reply is None when no built-in command matched and the
    caller should fall back to the LLM. wants_face_name is True
    when the next utterance should be used as a face name.
    """

    # Facts first, so "remember that my car is red" and "what is
    # my wifi password" never reach the camera code or qwen.
    reply = remember_fact(user_input)

    if reply is not None:
        return reply, False

    reply = recall_fact(user_input)

    if reply is not None:
        return reply, False

    intent, time_kind = detect_intent(user_input)

    if intent == "learn_face":

        return (
            f"Of course, {CALLNAME}. "
            "What is this person's name?"
        ), True

    if intent is not None:
        print(f"[FRIDAY] Understood as: {intent}")

    if intent == "time":
        return tell_time(time_kind), False

    if intent == "count":
        return describe_face_count(), False

    if intent == "who":
        return recognize_current_people(), False

    if intent == "see":
        return describe_scene(), False

    return None, False
