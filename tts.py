import os
import time

from collections import deque
from queue import Queue
from threading import Thread, Event, Lock

from config import TTS_MODEL_DIR, TTS_LANGUAGE, TTS_VOICE, TTS_DEVICE

# Keep the Pocket TTS weights inside the project (models/pocket_tts)
# instead of ~/.cache. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_CACHE", TTS_MODEL_DIR)

# Everything is downloaded already (see README), so never touch the
# network at runtime. Set FRIDAY_TTS_ONLINE=1 to allow downloads.
if os.environ.get("FRIDAY_TTS_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import sounddevice as sd
import torch

from pocket_tts import TTSModel

# Minimum seconds of audio generated before playback starts.
PREBUFFER_SECONDS = 0.25

# Rough speaking rate, used to guess how long a reply will be
# before it is generated (about 14 characters per second).
SECONDS_PER_CHAR = 0.07

# Extra audio kept in hand on top of the estimate, in seconds.
SAFETY_SECONDS = 0.3

# Seconds of audio the sound card / PulseAudio keeps buffered.
OUTPUT_LATENCY = 0.3

# Extra delay of the Bluetooth speaker itself, waited out before
# speak() returns.
BLUETOOTH_DELAY = 0.15

# Never wait longer than this much audio before starting to speak,
# even if generation is slower than real time. Keeps the start
# snappy like Piper; the rest streams in while it talks.
MAX_PREBUFFER_SECONDS = 0.6


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
                # Generation fell behind mid-reply.
                self.gaps += 1

    def close(self):

        if self.stream is not None:
            self.stream.abort()
            self.stream.close()
            self.stream = None


def load_voice():

    print("[FRIDAY] Loading Pocket TTS...")

    model = TTSModel.load_model(language=TTS_LANGUAGE)

    device = TTS_DEVICE

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[FRIDAY] CUDA not available, Pocket TTS stays on the CPU.")
        device = "cpu"

    # Move the model before building the voice state, so the voice
    # state is created on the same device.
    model.to(device)

    voice_state = model.get_state_for_audio_prompt(TTS_VOICE)

    # The first generation is slower, so do it now instead of on
    # the first "Yes, sir?".
    model.generate_audio(voice_state, "Hello.")

    print(
        f"[FRIDAY] Pocket TTS ready "
        f"(voice: {TTS_VOICE}, device: {device})."
    )

    return model, voice_state


class TTSEngine:

    def __init__(self, voice, speaking_event=None):

        self.model, self.voice_state = voice

        # Optional Event that is set while audio is being generated
        # and played (used to pause the camera and YOLO).
        self.speaking_event = speaking_event

        self.sample_rate = self.model.sample_rate

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

        # Audio is played chunk by chunk while the rest is still
        # being generated, so speech starts sooner. Playback waits
        # until the audio already generated covers the estimated time
        # still needed for the rest, but never longer than
        # MAX_PREBUFFER_SECONDS.
        player = self.player

        player.start()
        player.take_gaps()

        pending = []
        playing = False
        buffered = 0
        min_prebuffer = int(self.sample_rate * PREBUFFER_SECONDS)
        expected_seconds = max(0.5, len(text) * SECONDS_PER_CHAR)
        started = time.monotonic()

        try:

            # Don't break out of this loop: Pocket TTS ends the
            # stream itself once stop_event is set, and leaving early
            # skips its thread cleanup (crashes Python on exit).
            for chunk in self.model.generate_audio_stream(
                self.voice_state,
                text,
                stop=self.stop_event
            ):

                if self.stop_event.is_set():
                    continue

                audio = (
                    chunk.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1)
                )

                buffered += len(audio)

                if playing:
                    player.add(audio)
                    continue

                pending.append(audio)

                if self.ready_to_play(
                    buffered,
                    min_prebuffer,
                    expected_seconds,
                    time.monotonic() - started
                ):
                    player.add(np.concatenate(pending))
                    pending = []
                    playing = True

            if self.stop_event.is_set():
                return

            self.report_speed(buffered, time.monotonic() - started)

            # Short replies can end before playback has started.
            if pending:
                player.add(np.concatenate(pending))

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

    def ready_to_play(self, buffered, min_prebuffer, expected, elapsed):

        if buffered < min_prebuffer:
            return False

        buffered_seconds = buffered / self.sample_rate

        if buffered_seconds >= MAX_PREBUFFER_SECONDS:
            return True

        # Seconds of audio generated per second of wall time.
        speed = buffered_seconds / max(elapsed, 1e-3)

        remaining_audio = max(0.0, expected - buffered_seconds)

        # Time still needed to generate the rest at this speed.
        remaining_time = remaining_audio / max(speed, 1e-3)

        # While the buffered audio plays, the rest is generated.
        # Start only if that finishes before the buffer runs out.
        return buffered_seconds >= remaining_time + SAFETY_SECONDS

    def report_speed(self, samples, elapsed):

        # Real-time factor: above 1.0 means audio is generated
        # faster than it plays (smooth). Below 1.0 it can't keep up
        # and speech will break up: try the other TTS_DEVICE.
        audio_seconds = samples / self.sample_rate

        if audio_seconds <= 0:
            return

        print(
            f"[FRIDAY] TTS: {audio_seconds:.1f}s of audio in "
            f"{elapsed:.1f}s ({audio_seconds / max(elapsed, 1e-3):.2f}x "
            f"real time)"
        )

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
