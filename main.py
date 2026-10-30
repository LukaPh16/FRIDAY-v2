import re

from threading import Thread

import pyaudio

from arduino import Arduino
from audio_input import (
    load_whisper,
    find_mic,
    open_mic_stream,
    read_chunk,
    get_db,
    record_until_silence,
    transcribe
)
from camera import (
    start_face_detection,
    stop_face_detection,
    register_person_by_name
)
from commands import route_command
from config import NAME, CALLNAME, WAKEWORD, MIN_VOLUME
from llm import warmup_ollama, ask_ai
from objects import (
    load_object_detector,
    start_object_detection,
    stop_object_detection
)
from tts import load_voice, TTSEngine

GOODBYE_WORDS = ["goodbye", "bye", "exit"]
CANCEL_WORDS = ["cancel", "never mind", "nevermind"]


class Resources:
    """
    Everything cleanup() has to release. Fields stay None until
    their startup step succeeds, so a failed startup only cleans
    up what was actually created.
    """

    def __init__(self):

        self.model = None
        self.arduino = None
        self.p = None
        self.stream = None
        self.tts = None
        self.camera_thread = None
        self.object_thread = None


def start_up(res):

    res.model = load_whisper()

    voice = load_voice()

    object_detection_ready = load_object_detector()

    res.arduino = Arduino()

    # One PyAudio instance is used for finding the mic and
    # opening the stream, and is terminated in cleanup().
    res.p = pyaudio.PyAudio()

    device_index = find_mic(res.p)

    res.stream = open_mic_stream(res.p, device_index)

    print("[FRIDAY] Listening...")

    warmup_ollama()

    res.tts = TTSEngine(voice)

    res.camera_thread = Thread(
        target=start_face_detection,
        daemon=True
    )

    res.camera_thread.start()

    # Without YOLO, FRIDAY keeps working and says detection
    # isn't available.
    if object_detection_ready:

        res.object_thread = Thread(
            target=start_object_detection,
            daemon=True
        )

        res.object_thread.start()


def cleanup(res):

    print("\n[FRIDAY] Cleaning up resources...")

    # Stop object detection first: it reads camera frames.
    if res.object_thread is not None:
        try:
            stop_object_detection()

            if res.object_thread.is_alive():
                res.object_thread.join(timeout=3)

                if res.object_thread.is_alive():
                    print(
                        "[FRIDAY] Warning: Object detection "
                        "thread has not stopped yet."
                    )
                else:
                    print("[FRIDAY] Object detection thread stopped.")

        except Exception as e:
            print("[FRIDAY] Object detection cleanup error:", e)

    # 1. Stop the camera thread.
    if res.camera_thread is not None:
        try:
            stop_face_detection()

            if res.camera_thread.is_alive():
                res.camera_thread.join(timeout=3)

                if res.camera_thread.is_alive():
                    print(
                        "[FRIDAY] Warning: Camera thread "
                        "has not stopped yet."
                    )
                else:
                    print("[FRIDAY] Camera thread stopped.")

        except Exception as e:
            print("[FRIDAY] Camera cleanup error:", e)

    if res.tts is not None:

        # 2. Stop any remaining audio playback.
        try:
            res.tts.stop_playback()
        except Exception as e:
            print("[FRIDAY] Audio playback cleanup:", e)

        # 3. Stop the TTS worker.
        try:
            res.tts.shutdown()
        except Exception as e:
            print("[FRIDAY] TTS cleanup:", e)

    # 4. Close the microphone.
    if res.stream is not None:
        try:
            if res.stream.is_active():
                res.stream.stop_stream()

            res.stream.close()

            print("[FRIDAY] Microphone closed.")

        except Exception as e:
            print("[FRIDAY] Microphone cleanup error:", e)

    # 5. Release PyAudio.
    if res.p is not None:
        try:
            res.p.terminate()
            print("[FRIDAY] PyAudio released.")
        except Exception as e:
            print("[FRIDAY] PyAudio cleanup error:", e)

    # 6. Turn off and close Arduino communication.
    if res.arduino is not None:
        try:
            if res.arduino.is_open:
                res.arduino.set_mode("OFF")
                res.arduino.close()

            print("[FRIDAY] Arduino disconnected.")

        except Exception as e:
            print("[FRIDAY] Arduino cleanup error:", e)

    print("[FRIDAY] Cleanup finished.")


def go_idle(arduino, mode):

    if mode == "WAKE":
        arduino.set_mode("SLEEP")
    else:
        arduino.set_mode("IDLE")


def main(model, arduino, stream, tts):

    mode = "WAKE"
    waiting_for_command = False

    # Voice-controlled face registration
    pending_face_name = False

    arduino.set_mode("SLEEP")

    print("[FRIDAY] Main system started.")

    while True:

        data = read_chunk(stream)

        db = get_db(data)

        print(
            f"Volume: {db:6.2f} dB   ",
            end="\r",
            flush=True
        )

        if db <= MIN_VOLUME:
            continue

        arduino.set_mode("LISTEN")

        print("\n[FRIDAY] Speaking detected...")

        audio_data = record_until_silence(stream)

        arduino.set_mode("THINK")

        print("[FRIDAY] Processing...")

        raw_input = transcribe(model, audio_data)

        # Strip punctuation in every mode so commands like
        # "Goodbye." match the same way they do in WAKE mode.
        user_input = re.sub(
            r"[.!?,]",
            "",
            raw_input.lower()
        ).strip()

        print("[You]:", raw_input)

        if pending_face_name:

            if user_input in CANCEL_WORDS:

                pending_face_name = False

                reply = (
                    f"Face registration cancelled, {CALLNAME}."
                )

                print(f"{NAME}: {reply}")

                arduino.set_mode("TALK")

                tts.speak(reply)

                go_idle(arduino, mode)

                continue

            if (
                user_input in GOODBYE_WORDS
                or "be quiet" in user_input
                or "always listen" in user_input
            ):

                # This is a command, not a name: drop the
                # registration and handle it normally below.
                pending_face_name = False

        if pending_face_name:

            person_name = raw_input.strip(
                " .,!?:;"
            )

            if not person_name:

                reply = (
                    f"I didn't catch the name, {CALLNAME}. "
                    "Please say it again."
                )

                arduino.set_mode("TALK")

                tts.speak(reply)

                # Keep waiting for a name.
                go_idle(arduino, mode)

                continue

            print(
                f"[FRIDAY] Registering face for "
                f"{person_name}..."
            )

            success, result_message = register_person_by_name(
                person_name,
                timeout=10
            )

            pending_face_name = False

            if success:

                reply = (
                    f"I have saved the face as "
                    f"{result_message}, {CALLNAME}."
                )

            else:

                print(
                    f"[FRIDAY] Registration failed: "
                    f"{result_message}"
                )

                reply = (
                    f"I couldn't save the face, {CALLNAME}. "
                    "Please ensure one person's face is "
                    "visible and try again."
                )

            print(f"{NAME}: {reply}")

            arduino.set_mode("TALK")

            tts.speak(reply)

            go_idle(arduino, mode)

            continue

        if "be quiet" in user_input:

            mode = "WAKE"

            waiting_for_command = False

            print(
                f"[FRIDAY] Quiet mode enabled, "
                f"{CALLNAME}."
            )

            arduino.set_mode("TALK")

            tts.speak(
                f"Quiet mode enabled, {CALLNAME}."
            )

            arduino.set_mode("SLEEP")

            continue

        if "always listen" in user_input:

            mode = "ALWAYS"

            print(
                f"[FRIDAY] Always listening mode enabled, "
                f"{CALLNAME}."
            )

            arduino.set_mode("TALK")

            tts.speak(
                f"Always listening mode enabled, "
                f"{CALLNAME}."
            )

            arduino.set_mode("IDLE")

            continue

        if mode == "WAKE":

            if waiting_for_command:

                waiting_for_command = False

            else:

                if user_input == WAKEWORD:

                    print(f"[FRIDAY]: Yes, {CALLNAME}?")

                    arduino.set_mode("TALK")

                    tts.speak(
                        f"Yes, {CALLNAME}?"
                    )

                    arduino.set_mode("LISTEN")

                    waiting_for_command = True

                    continue

                elif user_input.startswith(WAKEWORD):

                    user_input = user_input.replace(
                        WAKEWORD,
                        "",
                        1
                    ).strip()

                    if not user_input:
                        continue

                else:

                    arduino.set_mode("SLEEP")

                    continue

        print("[You said]:", user_input)

        if not user_input:

            go_idle(arduino, mode)

            continue

        if user_input in GOODBYE_WORDS:

            reply = f"Goodbye, {CALLNAME}!"

            print(f"{NAME}: {reply}")

            arduino.set_mode("TALK")

            # Wait for the goodbye message to finish playing.
            tts.speak(reply)

            print("[FRIDAY] Goodbye message finished.")

            arduino.set_mode("OFF")

            return

        reply, pending_face_name = route_command(user_input)

        if reply is None:

            arduino.set_mode("THINK")

            reply = ask_ai(user_input)

        print(f"{NAME}: {reply}")

        arduino.set_mode("TALK")

        tts.speak(reply)

        go_idle(arduino, mode)


if __name__ == "__main__":

    res = Resources()

    try:
        start_up(res)

        main(res.model, res.arduino, res.stream, res.tts)

    except KeyboardInterrupt:
        print("\n[FRIDAY] Interrupted by user.")

    except Exception as e:
        print(f"\n[FRIDAY] Unexpected error: {e}")
        raise

    finally:
        cleanup(res)

        print("[FRIDAY] Shutdown complete.")
