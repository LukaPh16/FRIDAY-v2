import datetime

from camera import get_face_count, get_face_results
from config import CALLNAME
from memory import remember_fact, recall_fact
from objects import describe_scene


def tell_time(text):

    text = text.lower()

    now = datetime.datetime.now()

    if " time " in text:

        return now.strftime(
            f"The time is %H:%M, {CALLNAME}"
        )

    if " date " in text:

        return now.strftime(
            f"Today is %B %d, {CALLNAME}"
        )

    if " year " in text:

        return now.strftime(
            f"It is %Y, {CALLNAME}"
        )

    if " day " in text:

        return now.strftime(
            f"Today is %A, {CALLNAME}"
        )

    return None


def is_face_registration_command(text):

    phrases = [
        "remember this face",
        "remember the face",
        "remember this person",
        "save this face",
        "save this person",
        "learn this face",
        "learn this person"
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def is_face_recognition_command(text):

    phrases = [
        "do you recognize this person",
        "do you recognize this face",
        "do you recognize these people",
        "do you recognize these faces",
        "do you recognise this person",
        "do you recognise this face",
        "do you recognise these people",
        "do you recognise these faces",
        "who is this person",
        "who is this",
        "who are these people",
        "who are these",
        "who do you see",
        "identify this person",
        "identify this face",
        "identify these people",
        "identify these faces"
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def is_face_count_command(text):

    return (
        "how many people do you see"
        in text
        or "how many faces do you see"
        in text
    )


def is_scene_command(text):

    phrases = [
        "what do you see",
        "what can you see",
        "what's in front of you",
        "what is in front of you",
        "look around",
        "describe what you see"
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


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
    Match user input against the built-in commands.

    Returns:
        (reply, wants_face_name)

    reply is None when no built-in command matched and the
    caller should fall back to the LLM. wants_face_name is True
    when the next utterance should be used as a face name.
    """

    if is_face_registration_command(user_input):

        return (
            f"Of course, {CALLNAME}. "
            "What is this person's name?"
        ), True

    reply = None

    if is_face_recognition_command(user_input):

        reply = recognize_current_people()

    if reply is None:

        reply = remember_fact(user_input)

    if reply is None:

        reply = recall_fact(user_input)

    if reply is None:
        padded_input = f" {user_input} "

        reply = tell_time(padded_input)

    if reply is None:

        if is_face_count_command(user_input):

            reply = describe_face_count()

    if reply is None:

        if is_scene_command(user_input):

            reply = describe_scene()

    return reply, False
