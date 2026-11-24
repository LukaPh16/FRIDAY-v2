# FRIDAY v2

A voice assistant I'm building on an NVIDIA Jetson, loosely based on FRIDAY from Iron Man. Everything runs locally: speech recognition, the LLM, text to speech, face recognition and object detection. No cloud APIs.

Still a work in progress, so expect things to change.

## What it can do

- Listens for the wake word "Friday", or always listens if you tell it to
- Answers questions with a small local LLM (qwen2.5:0.5b through Ollama)
- Talks back with Piper, voice "amy"
- Learns faces by voice and recognizes people later
- Describes what it sees with YOLO11n, e.g. "I see Luka, a laptop and two cups, sir."
- Remembers simple facts you tell it
- Tells the time and date
- Sends its current state (SLEEP, LISTEN, THINK, TALK, IDLE, OFF) to an Arduino over serial

## Hardware

- NVIDIA Jetson (JetPack, Python 3.10)
- USB microphone
- USB webcam
- Speaker
- Arduino on /dev/ttyACM0

## Commands

By default it only reacts after you say "Friday". You can say "Friday" (or "hey Friday") and wait for "Yes, sir?", or say it all at once: "Friday, what time is it".

You don't have to use exact wording. The examples below are just examples, and other phrasings with the same meaning work too ("got the time?", "what can you spot over there", "how many folks are here", "who's that standing there"). `intent.py` works out what you mean from the words you use. When a sentence is borderline (e.g. "what is this"), it asks qwen to decide between the likely commands and normal conversation.

| Say something like | Does |
|---|---|
| always listen / keep listening / stay awake | stops needing the wake word |
| be quiet / go to sleep / stop listening | goes back to wake word mode |
| what's the time / what's today's date / what day is it / what year is it | time and date |
| remember my favorite color is red / my keys are in the drawer, remember that | saves a fact |
| what's my favorite color / where are my keys / recall my favorite color | says the saved fact |
| remember this face / learn this person / meet my friend | asks for a name ("his name is John" works) and saves the face. "cancel" or "never mind" to stop |
| who is this / do you know who that is / do you recognize me | says who it sees |
| how many people do you see / how many people are here | counts faces |
| what do you see / what's in front of you / look around | describes people and objects in front of it |
| goodbye / bye Friday / shut down | shuts down |

Only "always listen" and "be quiet" work without the wake word in wake word mode. The other mode phrases need "Friday" first, so background talk doesn't switch modes.

Anything else goes to the LLM. It sees the last few exchanges and your saved facts, so follow-up questions work ("who wrote Harry Potter?" then "when was she born?").

## Setup

Create a venv and install NVIDIA's Jetson PyTorch wheel first (the normal pip version runs on the CPU only). It has to match the JetPack version: JetPack 6.0 ships cuDNN 8, so use the 2.4.0 build from NVIDIA's `jp/v60` folder. The 2.5.0 build is for JetPack 6.1 and needs cuDNN 9.

```
python3 -m venv venv
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
venv/bin/pip install torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl
```

This torch also needs cuSPARSELt (the `cusparselt-local-tegra-repo` .deb), which installs into a folder the linker doesn't search. Add it once:

```
echo /usr/lib/aarch64-linux-gnu/libcusparseLt/12 | sudo tee /etc/ld.so.conf.d/cusparselt.conf
sudo ldconfig
```

NVIDIA's torch 2.4 is built against numpy 1.x, so numpy has to stay below 2 (and opencv-python below 4.12, which needs numpy 2). Install Piper with `--no-deps` so pip leaves onnxruntime alone:

```
venv/bin/pip install --no-deps piper-tts
venv/bin/pip install "numpy<2" "opencv-python<4.12" pathvalidate scipy soundfile openai-whisper ollama pyaudio sounddevice pyserial onnxruntime requests
```

Install Ollama and pull the model:

```
ollama pull qwen2.5:0.5b
```

The model files aren't in the repo, so you need to download them and put them here:

```
FRIDAY-v2/
  models/
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
    yolo11n.onnx
    piper/
      en_US-amy-medium.onnx
      en_US-amy-medium.onnx.json
```

- Piper Amy voice: https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0/en/en_US/amy/medium (the voice is set with `TTS_MODEL` in config.py)

- YuNet and SFace: https://github.com/opencv/opencv_zoo
- YOLO11n: exported from Ultralytics with `yolo export model=yolo11n.pt format=onnx`

If yolo11n.onnx is missing it still runs, it just says object detection isn't available.

Then run:

```
venv/bin/python main.py
```

Ctrl+C or "goodbye" to stop. memory.json and faces.json get created on their own.

Settings like the wake word, what it calls you, mic name and sensitivity, and the Arduino port are all in config.py.

## Files

- main.py: startup, the main loop and cleanup
- config.py: settings
- audio_input.py: mic, recording and Whisper
- tts.py: Piper voice output
- llm.py: Ollama
- commands.py: the built-in commands
- memory.py: remember / recall
- camera.py: webcam and face recognition
- objects.py: YOLO object detection
- arduino.py: serial connection to the Arduino
- test_describe_scene.py: test for the "what do you see" sentences, no hardware needed
- camera_index_check.py: finds which camera index works
- drop_caches.sh: frees up RAM on the Jetson

## To do

- Conversation memory for the LLM
- Longer replies (they get cut off right now)
- More flexible wake word, like "Hey Friday"
- Better name handling when saving faces
- requirements.txt
- Upload the Arduino code