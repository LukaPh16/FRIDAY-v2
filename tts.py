import time

from collections import deque
from queue import Queue
from threading import Thread, Event, Lock

import numpy as np
import sounddevice as sd

from piper import PiperVoice

from config import TTS_MODEL, TTS_CONFIG

# Seconds of audio the sound card / PulseAudio keeps buffered.
OUTPUT_LATENCY = 0.3

# Extra delay of the Bluetooth speaker itself, waited out before
# speak() returns.
BLUETOOTH_DELAY = 0.15


class AudioPlayer:
    """
    One output stream that stays open for FRIDAY's whole run and
    plays silence when there is nothing to say.

    The speaker is Bluetooth (via PulseAudio). PulseAudio suspends
    an idle Bluetooth sink, and opening a new stream per reply wakes
    it up, which made speech stall after the first few words. Keeping
    the stream open keeps the sink awake.

    The PortAudio callback pulls audio from a queue on its own
    thread, so a late chunk plays as silence instead of starving
    ALSA (underrun).
    """

    def __init__(self, sample_rate):

        self.sample_rate = sample_rate
        self.items = deque()
        self.offset = 0
        self.lock = Lock()
        self.stream = None

        # True while a reply is being played; used to count gaps.
        self.active = False
        self.gaps = 0

    def start(self):

        if self.stream is not None:
            return

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self.callback,
            # latency="high" is only ~35 ms through PulseAudio,
            # far too small for Bluetooth.
            latency=OUTPUT_LATENCY
        )

        self.stream.start()

    def add(self, audio):

        with self.lock:
            self.items.append(audio)
            self.active = True

    def end(self):
        """
        Mark the end of a reply. Returns an Event that is set once
        all of the reply has been handed to the sound card.
        """

        marker = Event()

        with self.lock:
            self.items.append(marker)

        return marker

    def clear(self):
        """Drop everything queued (used when speech is interrupted)."""

        with self.lock:

            for item in self.items:
                if isinstance(item, Event):
                    item.set()

            self.items.clear()
            self.offset = 0
            self.active = False

    def take_gaps(self):

        with self.lock:
            gaps = self.gaps
            self.gaps = 0

        return gaps

    def callback(self, outdata, frames, time_info, status):

        out = outdata[:, 0]
        filled = 0

        with self.lock:

            while filled < frames and self.items:

                item = self.items[0]

                if isinstance(item, Event):
                    # End of a reply.
                    self.items.popleft()
                    self.active = False
                    item.set()
                    continue

                n = min(frames - filled, len(item) - self.offset)

                out[filled:filled + n] = item[self.offset:self.offset + n]

                filled += n
                self.offset += n

                if self.offset >= len(item):
                    self.items.popleft()
                    self.offset = 0

            # Silence for whatever isn't filled.
            out[filled:] = 0

            if self.active and filled < frames:
                # The next sentence wasn't ready yet.
                self.gaps += 1

    def close(self):

        if self.stream is not None:
            self.stream.abort()
            self.stream.close()
            self.stream = None


def load_voice():

    print("[FRIDAY] Loading Piper...")

    voice = PiperVoice.load(
        TTS_MODEL,
        config_path=TTS_CONFIG
    )

    # The first synthesis is slower, so do it now instead of on
    # the first "Yes, sir?".
    for _ in voice.synthesize("Hello."):
        pass

    print(f"[FRIDAY] Piper ready ({TTS_MODEL}).")

    return voice


class TTSEngine:

    def __init__(self, voice, speaking_event=None):

        self.voice = voice

        # Optional Event that is set while audio is being generated
        # and played (used to pause the camera and YOLO).
        self.speaking_event = speaking_event

        self.sample_rate = voice.config.sample_rate

        self.player = AudioPlayer(self.sample_rate)

        try:
            self.player.start()
        except Exception as e:
            # Retried on the next reply.
            print("[FRIDAY] Could not open audio output:", e)

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

    def play_stream(self, text):

        # Piper synthesizes one sentence at a time. Each sentence is
        # played as soon as it's ready while the next one is being
        # synthesized, so long replies start right away.
        player = self.player

        player.start()
        player.take_gaps()

        try:

            for chunk in self.voice.synthesize(text):

                if self.stop_event.is_set():
                    return

                player.add(
                    chunk.audio_float_array
                    .astype(np.float32)
                    .reshape(-1)
                )

            marker = player.end()

            # Wait until the whole reply is handed to the sound card,
            # or for an interruption.
            while not marker.wait(timeout=0.05):

                if self.stop_event.is_set():
                    return

            # The sound card and Bluetooth still hold the last part,
            # so wait for it to be heard before returning (otherwise
            # the mic can pick up FRIDAY's own voice).
            time.sleep(player.stream.latency + BLUETOOTH_DELAY)

            gaps = player.take_gaps()

            if gaps:
                print(
                    f"[FRIDAY] TTS: audio ran out {gaps} "
                    f"time(s) while speaking."
                )

        finally:

            if self.stop_event.is_set():
                # Interrupted: drop what's still queued.
                player.clear()

    def worker(self):

        while True:

            text, done = self.queue.get()

            try:

                # Shutdown signal
                if text is None:
                    return

                self.stop_event.clear()

                if self.speaking_event is not None:
                    self.speaking_event.set()

                self.play_stream(text)

            except Exception as e:

                print("[FRIDAY] TTS error:", e)

            finally:

                if self.speaking_event is not None:
                    self.speaking_event.clear()

                done.set()

                self.queue.task_done()

    def stop_playback(self):

        self.stop_event.set()

    def shutdown(self):

        done = Event()

        self.queue.put((None, done))

        done.wait(timeout=3)

        self.thread.join(timeout=3)

        self.player.close()

        print("[FRIDAY] TTS worker stopped.")
