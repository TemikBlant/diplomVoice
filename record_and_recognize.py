import pyaudio
import wave
from recognizer import Recognizer
import sys

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 128
RECORD_SECONDS = 1
device_index = 2
audio = pyaudio.PyAudio()
r = Recognizer()
r.load_models()


print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")

index = int(input())
print("recording via index " + str(index))


flag = True
while flag:
    command_start = str(input())
    if command_start == 'q':
        flag = False
        sys.exit()
    print("Ready")
    print("recording started")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    stream.stop_stream()
    stream.close()
    waveFile = wave.open("temp\\test.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    print(r.recognize(), end='')
    print("recording stopped")
audio.terminate()
