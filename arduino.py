import time

import serial

from config import ARDUINO_PORT, ARDUINO_BAUD


class Arduino:

    def __init__(self, port=ARDUINO_PORT, baud=ARDUINO_BAUD):

        self.serial = serial.Serial(
            port,
            baud,
            timeout=1
        )

        time.sleep(2)

        print("[FRIDAY] Connected to Arduino.")

    def send(self, command):

        self.serial.write(
            (command + "\n").encode()
        )

    def set_mode(self, mode):

        try:

            self.send(mode)

        except Exception as e:

            print("[FRIDAY] Arduino error:", e)

    @property
    def is_open(self):

        return self.serial.is_open

    def close(self):

        self.serial.close()
