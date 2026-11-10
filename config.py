import os

NAME = "FRIDAY"
CALLNAME = "sir"

WAKEWORD = "friday"

# Audio
RATE = 48000
TARGET_RATE = 16000
CHUNK_SIZE = 1024

MIN_VOLUME = 65
SILENCE_THRESHOLD = 60
SILENCE_LIMIT = 20

# Stop recording after this long even if it never goes quiet.
MAX_RECORD_SECONDS = 15

MIC_NAME = "USB Composite Device: Audio"

# Models
WHISPER_MODEL = "base"

# Pocket TTS (Kyutai). Weights and the voice live in models/pocket_tts.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TTS_MODEL_DIR = os.path.join(_BASE_DIR, "models", "pocket_tts")

# "english_7m_2026-10" is the small model: faster than real time on
# the Orin Nano CPU. "english" is the full model: a bit better quality
# but about 2x slower than real time, so streamed speech stutters.
TTS_LANGUAGE = "english_7m_2026-10"

# A built-in voice name, or a path to a .wav / exported .safetensors.
TTS_VOICE = "eve"

# "cpu" or "cuda". Measured on the Orin Nano with english_7m:
# CPU 2.4x real time, first audio after 0.12s; GPU only 1.4x and
# 0.25s (the model is too small for the GPU to pay off). Use "cuda"
# only with the full "english" model, which is too slow on the CPU.
TTS_DEVICE = "cpu"

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_KEEP_ALIVE = "30m"

# Arduino
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200

# Memory
MEMORY_FILE = "memory.json"

# Object detection
OBJECT_DETECTION_FPS = 3
