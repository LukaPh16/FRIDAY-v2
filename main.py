import sys
import whisper
import pyaudio
import numpy as np
import audioop
import threading
import time
import ollama

from queue import Queue
from threading import Thread, Event

from piper.voice import PiperVoice
import sounddevice as sd
import soundfile as sf
import wave

import io
import json
import re

import datetime
import serial

import os
import requests

from camera import (
    start_face_detection,
    stop_face_detection,
    get_face_count,
    get_recognized_people,
    get_face_results,
    register_person_by_name
)

LAPTOP_IP = "10.247.248.44"

NAME = "FRIDAY"
CALLNAME = "SIR"

WAKEWORD = "friday"
MODE = "WAKE"
waiting_for_command = False

# Camera
camera_mode = False
camera_thread = None

# Voice-controlled face registration
pending_face_name = False

# Audio
RATE = 48000
TARGET_RATE = 16000
CHUNK_SIZE = 1024

MIN_VOLUME = 65
SILENCE_THRESHOLD = 60
SILENCE_LIMIT = 20

# Models
TTS_MODEL = "en_US-hfc_female-medium.onnx"
TTS_CONFIG = "en_US-hfc_female-medium.onnx.json"

OLLAMA_MODEL = "qwen2.5:0.5b"

# Arduino
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200

# Memory
MEMORY_FILE = "memory.json"

print("[FRIDAY] Loading Whisper...")

model = whisper.load_model(
    "base",
    device="cuda"
)

print("[FRIDAY] Loading Piper...")

voice = PiperVoice.load(
    TTS_MODEL,
    config_path=TTS_CONFIG
)

arduino = serial.Serial(
    ARDUINO_PORT,
    ARDUINO_BAUD,
    timeout=1
)

time.sleep(2)

print("[FRIDAY] Connected to Arduino.")

def find_mic():

    p = pyaudio.PyAudio()

    target = "USB Composite Device: Audio"

    for i in range(p.get_device_count()):

        info = p.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:

            name = info["name"]

            if target.lower() in name.lower():

                print(f"[FRIDAY] Using microphone: {name}")

                return i

    raise Exception("USB microphone not found!")


DEVICE_INDEX = find_mic()

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK_SIZE
)

print("[FRIDAY] Listening...")

tts_stop_event = Event()

tts_playing = False

tts_lock = threading.Lock()

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
        }
    )

    print("[FRIDAY] Ollama ready!")


os.environ["OLLAMA_KEEP_ALIVE"] = "30m"

warmup_ollama()

def get_db(block):

    data = np.frombuffer(
        block,
        dtype=np.int16
    ).astype(np.float32)

    rms = np.sqrt(np.mean(data ** 2))

    return 20 * np.log10(rms) if rms > 0 else -100

class TTSEngine:

    def __init__(self, voice):

        self.voice = voice

        self.queue = Queue()

        self.running = True

        self.thread = Thread(
            target=self.worker,
            daemon=True
        )

        self.thread.start()

    def speak(self, text):

        if not text:
            return

        done = Event()

        self.queue.put((text, done))

        done.wait()

    def worker(self):

        global tts_playing

        while True:

            text, done = self.queue.get()

            try:

                # Shutdown signal
                if text is None:
                    done.set()
                    return

                buffer = io.BytesIO()

                with wave.open(buffer, "wb") as f:

                    self.voice.synthesize_wav(
                        text,
                        f
                    )

                buffer.seek(0)

                data, fs = sf.read(buffer)

                tts_playing = True

                sd.play(data, fs)

                while sd.get_stream().active:

                    if tts_stop_event.is_set():

                        sd.stop()

                        tts_stop_event.clear()

                        break

                    time.sleep(0.01)

            except Exception as e:

                print("[FRIDAY] TTS error:", e)

            finally:

                tts_playing = False

                done.set()

                self.queue.task_done()
    
    def shutdown(self):

        self.running = False

        done = Event()

        self.queue.put((None, done))

        done.wait(timeout=3)

        self.thread.join(timeout=3)

        print("[FRIDAY] TTS worker stopped.")

tts = TTSEngine(voice)

def cleanup():

    global camera_thread

    print("\n[FRIDAY] Cleaning up resources...")

    # 1. Stop the camera thread.
    try:
        stop_face_detection()

        if (
            camera_thread is not None
            and camera_thread.is_alive()
        ):
            camera_thread.join(timeout=3)

            if camera_thread.is_alive():
                print(
                    "[FRIDAY] Warning: Camera thread "
                    "has not stopped yet."
                )
            else:
                print("[FRIDAY] Camera thread stopped.")

    except Exception as e:
        print("[FRIDAY] Camera cleanup error:", e)

    # 2. Stop any remaining audio playback.
    try:
        tts_stop_event.set()
        sd.stop()
    except Exception as e:
        print("[FRIDAY] Audio playback cleanup:", e)

    # 3. Stop the TTS worker.
    try:
        tts.shutdown()
    except Exception as e:
        print("[FRIDAY] TTS cleanup:", e)

    # 4. Close the microphone.
    try:
        if stream.is_active():
            stream.stop_stream()

        stream.close()

        print("[FRIDAY] Microphone closed.")

    except Exception as e:
        print("[FRIDAY] Microphone cleanup error:", e)

    # 5. Release PyAudio.
    try:
        p.terminate()
        print("[FRIDAY] PyAudio released.")
    except Exception as e:
        print("[FRIDAY] PyAudio cleanup error:", e)

    # 6. Turn off and close Arduino communication.
    try:
        if arduino.is_open:
            set_mode("OFF")
            arduino.close()

        print("[FRIDAY] Arduino disconnected.")

    except Exception as e:
        print("[FRIDAY] Arduino cleanup error:", e)

    print("[FRIDAY] Cleanup finished.")

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


# ==========================================
# AI CONVERSATION
# ==========================================

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
        }
    )

    reply = response["message"]["content"]

    reply = limit_sentences(reply)

    reply = enforce_callname(reply)

    print(f"{NAME}: {reply}")

    return reply

def send(command):

    arduino.write(
        (command + "\n").encode()
    )


def set_mode(mode):

    try:

        send(mode)

    except Exception as e:

        print("[FRIDAY] Arduino error:", e)

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

            return f"{memory[key]}. {CALLNAME}."

        return f"I don't have that information, {CALLNAME}."

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

    if unknown_count == 1:
        parts.append(
            "I don't recognize one person"
        )

    elif unknown_count > 1:
        parts.append(
            f"I don't recognize {unknown_count} people"
        )

    if not known_people:
        parts.append(
            "I don't recognize anyone"
        )

    reply = ". ".join(parts) + f", {CALLNAME}."

    return reply

def send_laptop_command(command):

    try:

        requests.post(
            f"http://{LAPTOP_IP}:5000/command",
            json={
                "cmd": command
            },
            timeout=2
        )

    except Exception as e:

        print("[FRIDAY] Laptop error:", e)

def main():

    global examples
    global MODE
    global waiting_for_command
    global camera_mode
    global camera_thread
    global pending_face_name

    camera_thread = Thread(
        target=start_face_detection,
        daemon=True
    )

    camera_thread.start()

    set_mode("SLEEP")

    print("[FRIDAY] Main system started.")

    while True:

        data = stream.read(
            CHUNK_SIZE,
            exception_on_overflow=False
        )

        db = get_db(data)

        print(
            f"Volume: {db:6.2f} dB   ",
            end="\r",
            flush=True
        )

        if db <= MIN_VOLUME:
            continue

        set_mode("LISTEN")

        print("\n[FRIDAY] Speaking detected...")

        frames = []

        silence_count = 0

        while True:

            chunk = stream.read(
                CHUNK_SIZE,
                exception_on_overflow=False
            )

            frames.append(chunk)

            db = get_db(chunk)

            if db < SILENCE_THRESHOLD:

                silence_count += 1

            else:

                silence_count = 0

            if silence_count > SILENCE_LIMIT:
                break

        set_mode("THINK")

        print("[FRIDAY] Processing...")

        audio_data = b"".join(frames)

        # Convert 48 kHz to 16 kHz.
        data_16k, _ = audioop.ratecv(
            audio_data,
            2,
            1,
            RATE,
            TARGET_RATE,
            None
        )

        audio_np = (
            np.frombuffer(
                data_16k,
                np.int16
            ).astype(np.float32) / 32768.0
        )

        result = model.transcribe(
            audio_np,
            language="en"
        )

        raw_input = result["text"].strip()

        user_input = raw_input.lower().strip()

        print("[You]:", raw_input)

        if pending_face_name:

            person_name = raw_input.strip(
                " .,!?:;"
            )

            if not person_name:

                reply = (
                    f"I didn't catch the name, {CALLNAME}. "
                    "Please say it again."
                )

                set_mode("TALK")

                tts.speak(reply)

                # Keep waiting for a name.
                pending_face_name = True

                if MODE == "WAKE":
                    set_mode("SLEEP")
                else:
                    set_mode("IDLE")

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

            set_mode("TALK")

            tts.speak(reply)

            if MODE == "WAKE":

                set_mode("SLEEP")

            else:

                set_mode("IDLE")

            continue

        if "be quiet" in user_input:

            MODE = "WAKE"

            waiting_for_command = False

            print(
                f"[FRIDAY] Quiet mode enabled, "
                f"{CALLNAME}."
            )

            set_mode("TALK")

            tts.speak(
                f"Quiet mode enabled, {CALLNAME}."
            )

            set_mode("SLEEP")

            continue

        if "always listen" in user_input:

            MODE = "ALWAYS"

            print(
                f"[FRIDAY] Always listening mode enabled, "
                f"{CALLNAME}."
            )

            set_mode("TALK")

            tts.speak(
                f"Always listening mode enabled, "
                f"{CALLNAME}."
            )

            set_mode("IDLE")

            continue

        if MODE == "WAKE":

            user_input = re.sub(
                r"[.!?,]",
                "",
                user_input
            ).strip()

            if waiting_for_command:

                waiting_for_command = False

            else:

                if user_input == WAKEWORD:

                    print(f"[FRIDAY]: Yes, {CALLNAME}?")

                    set_mode("TALK")

                    tts.speak(
                        f"Yes, {CALLNAME}?"
                    )

                    set_mode("LISTEN")

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

                    set_mode("SLEEP")

                    continue

        print("[You said]:", user_input)

        if not user_input:

            if MODE == "WAKE":
                set_mode("SLEEP")
            else:
                set_mode("IDLE")

            continue

        if user_input in ["goodbye", "bye", "exit"]:

            reply = f"Goodbye, {CALLNAME}!"

            print(f"{NAME}: {reply}")

            set_mode("TALK")

            # Wait for the goodbye message to finish playing.
            tts.speak(reply)

            print("[FRIDAY] Goodbye message finished.")

            set_mode("OFF")

            return

        if MODE not in ["WAKE", "ALWAYS"]:
            continue

        reply = None

        if is_face_registration_command(user_input):

            pending_face_name = True

            reply = (
                f"Of course, {CALLNAME}. "
                "What is this person's name?"
            )

        elif is_face_recognition_command(user_input):

            reply = recognize_current_people()

        if reply is None:

            reply = remember_fact(user_input)

        if reply is None:

            reply = recall_fact(user_input)

        if reply is None:
            padded_input = f" {user_input} "

            reply = tell_time(padded_input)

        if reply is None:

            if (
                "how many people do you see"
                in user_input
                or "how many faces do you see"
                in user_input
            ):

                count = get_face_count()

                if count == 1:

                    reply = (
                        f"I currently detect one face, "
                        f"{CALLNAME}."
                    )

                else:

                    reply = (
                        f"I currently detect {count} faces, "
                        f"{CALLNAME}."
                    )

        if reply is None:

            match = re.search(
                r"open (.+)",
                user_input
            )

            if match:

                app = match.group(1).strip().rstrip(".!?")

                send_laptop_command(
                    f"open {app}"
                )

                reply = (
                    f"Opening {app} on your laptop, "
                    f"{CALLNAME}."
                )

        if reply is None:

            set_mode("THINK")

            reply = ask_ai(user_input)

        print(f"{NAME}: {reply}")

        set_mode("TALK")

        tts.speak(reply)

        if MODE == "WAKE":

            set_mode("SLEEP")

        else:

            set_mode("IDLE")

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n[FRIDAY] Interrupted by user.")

    except Exception as e:
        print(f"\n[FRIDAY] Unexpected error: {e}")
        raise

    finally:
        cleanup()

        print("[FRIDAY] Shutdown complete.")