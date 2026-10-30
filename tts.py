import io
import time
import wave

from queue import Queue
from threading import Thread, Event

from piper.voice import PiperVoice
import sounddevice as sd
import soundfile as sf

from config import TTS_MODEL, TTS_CONFIG


def load_voice():

    print("[FRIDAY] Loading Piper...")

    return PiperVoice.load(
        TTS_MODEL,
        config_path=TTS_CONFIG
    )


class TTSEngine:

    def __init__(self, voice):

        self.voice = voice

        self.queue = Queue()

        self.stop_event = Event()

        self.thread = Thread(
            target=self.worker,
            daemon=True
        )

        self.thread.start()

    def speak(self, text):

        if not text:
            return

        done = Event()

        if not self.thread.is_alive():
            print("[FRIDAY] TTS worker is not running.")
            return

        self.queue.put((text, done))

        # Wait for playback to finish, but don't hang if the
        # worker thread has died.
        while not done.wait(timeout=0.5):

            if not self.thread.is_alive():
                print("[FRIDAY] TTS worker is not running.")
                return

    def worker(self):

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

                sd.play(data, fs)

                while sd.get_stream().active:

                    if self.stop_event.is_set():

                        sd.stop()

                        self.stop_event.clear()

                        break

                    time.sleep(0.01)

            except Exception as e:

                print("[FRIDAY] TTS error:", e)

            finally:

                done.set()

                self.queue.task_done()

    def stop_playback(self):

        self.stop_event.set()
        sd.stop()

    def shutdown(self):

        done = Event()

        self.queue.put((None, done))

        done.wait(timeout=3)

        self.thread.join(timeout=3)

        print("[FRIDAY] TTS worker stopped.")
