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

# Piper voice (Amy, medium quality). Files live in models/piper.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TTS_MODEL = os.path.join(
    _BASE_DIR, "models", "piper", "en_US-amy-medium.onnx"
)
TTS_CONFIG = TTS_MODEL + ".json"

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_KEEP_ALIVE = "30m"

# Arduino
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200

# Memory
MEMORY_FILE = "memory.json"

# Object detection
OBJECT_DETECTION_FPS = 3
