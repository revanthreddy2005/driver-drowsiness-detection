# create_alarm.py
import wave
import struct
import math

filename = "alarm.wav"
duration_s = 1.0       # 1 second
freq_hz = 880.0        # frequency of the beep (880 Hz = A5)
sample_rate = 44100
amplitude = 16000      # 16-bit audio

n_samples = int(sample_rate * duration_s)
wav_file = wave.open(filename, 'w')
wav_file.setparams((1, 2, sample_rate, n_samples, 'NONE', 'not compressed'))

for i in range(n_samples):
    t = i / sample_rate
    value = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
    data = struct.pack('<h', value)
    wav_file.writeframesraw(data)

wav_file.close()
print("WAV_CREATED", filename)
