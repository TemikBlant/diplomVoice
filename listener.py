import pyaudio
import math
import struct
import wave
import sys
from recognizer import Recognizer
import time


class Listener_micro:

    def __init__(self, treshold):
        self.Threshold = treshold
        self.SHORT_NORMALIZE = (1.0 / 32768.0)
        self.chunk = 128
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.swidth = 2
        self.Max_Seconds = 5
        self.seconds_of_record = 0.5
        self.timeout_signal = ((self.RATE / self.chunk * self.Max_Seconds) + 2)
        self.silence = True
        self.filename = 'temp/test.wav'
        self.Time = 0
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk,
                                  input_device_index=1)
        self.r = Recognizer()
        self.r.load_models()

    def get_stream(self, chunk):
        return self.stream.read(chunk)

    def rms(self, frame):
        count = len(frame) / self.swidth
        format = "%dh" % count
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * self.SHORT_NORMALIZE
            sum_squares += n * n
            rms = math.pow(sum_squares / count, 0.5)

            return rms * 1000

    def write_speech(self, write_data):
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(write_data)
        wf.close()
        self.r.recognize()

    def keep_record(self, last_block):
        all = [last_block]
        for i in range(0, int(self.RATE / self.chunk * self.seconds_of_record)):
            data = self.stream.read(self.chunk)
            all.append(data)
        data = b''.join(all)
        self.write_speech(data)
        silence = True
        Time = 0
        self.listen(silence, Time)

    def listen(self, silence=True, Time=0):
        print("waiting for Speech")
        kek = []
        while silence:
            try:
                input = self.get_stream(256)
                if len(kek) > 25:
                    for i in range(len(kek)):
                        kek[0] = kek[1]
                    kek[len(kek)-1] = input
                else:
                    kek.append(input)
            except:
                continue
            last_block = b''.join(kek)
            rms_value = self.rms(input)
            if rms_value > self.Threshold:
                silence = False
                print("Start record....")
                time.sleep(0)
                self.keep_record(last_block)
            Time = Time + 1
            if Time > self.timeout_signal:
                print("Time Out No Speech Detected")
                sys.exit()


l = Listener_micro(treshold=5)
l.listen()

