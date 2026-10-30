from math import gcd

import numpy as np
import pyaudio
import whisper

from scipy.signal import resample_poly

from config import (
    RATE,
    TARGET_RATE,
    CHUNK_SIZE,
    SILENCE_THRESHOLD,
    SILENCE_LIMIT,
    MAX_RECORD_SECONDS,
    MIC_NAME,
    WHISPER_MODEL
)


def load_whisper():

    print("[FRIDAY] Loading Whisper...")

    return whisper.load_model(
        WHISPER_MODEL,
        device="cuda"
    )


def find_mic(p):

    target = MIC_NAME

    for i in range(p.get_device_count()):

        info = p.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:

            name = info["name"]

            if target.lower() in name.lower():

                print(f"[FRIDAY] Using microphone: {name}")

                return i

    raise Exception("USB microphone not found!")


def open_mic_stream(p, device_index):

    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE
    )


def read_chunk(stream):

    return stream.read(
        CHUNK_SIZE,
        exception_on_overflow=False
    )


def get_db(block):

    data = np.frombuffer(
        block,
        dtype=np.int16
    ).astype(np.float32)

    rms = np.sqrt(np.mean(data ** 2))

    return 20 * np.log10(rms) if rms > 0 else -100


def record_until_silence(stream):

    max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK_SIZE)

    frames = []

    silence_count = 0

    while True:

        chunk = read_chunk(stream)

        frames.append(chunk)

        db = get_db(chunk)

        if db < SILENCE_THRESHOLD:

            silence_count += 1

        else:

            silence_count = 0

        if silence_count > SILENCE_LIMIT:
            break

        if len(frames) >= max_chunks:
            print("[FRIDAY] Maximum recording length reached.")
            break

    return b"".join(frames)


def resample_to_16k(audio_data):

    # Convert 48 kHz to 16 kHz.
    audio_np = (
        np.frombuffer(
            audio_data,
            np.int16
        ).astype(np.float32) / 32768.0
    )

    divisor = gcd(RATE, TARGET_RATE)

    return resample_poly(
        audio_np,
        TARGET_RATE // divisor,
        RATE // divisor
    ).astype(np.float32)


def transcribe(model, audio_data):

    audio_np = resample_to_16k(audio_data)

    result = model.transcribe(
        audio_np,
        language="en"
    )

    return result["text"].strip()
