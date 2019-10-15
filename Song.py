import wave
import scipy.io.wavfile as wavfile
from scipy.signal import decimate, butter, lfilter, freqz
from scipy.signal.windows import hamming
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np

# from pydub import AudioSegment
# from pydub.playback import play

DEFAULT_SAMPLING_RATE = 44100
SAMPLING_RATE = 11025
CUTOFF_FREQUENCY = 5000
SAMPLES_PER_WINDOW = 1024

def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# class Song:
#
#     def __init__(self, fileName, windowSize, samplingFrequency):
#         self.name = fileName
#         self.wSize = windowSize
#         self.fs = samplingFrequency


if __name__ == "__main__":
    # Read the .wav file
    rate, audio_data = wavfile.read('example.wav')
    print(rate, audio_data)

    # Convert stereo to mono
    audio_data = stereo_to_mono(audio_data)
    print(audio_data)
    first_len = len(audio_data)
    print(len(audio_data))

    # Pass the signal to a low-pass filter with a cutoff frequency of 5000 Hz
    filtered_data = butter_lowpass_filter(audio_data, CUTOFF_FREQUENCY, DEFAULT_SAMPLING_RATE)

    # Decimate by a factor of 4
    decimated_data = decimate(filtered_data, DEFAULT_SAMPLING_RATE / SAMPLING_RATE)
    print(decimated_data)
    second_len = len(audio_data)
    print(len(audio_data))
    assert (first_len / 4 == second_len)

    # Generate a hamming window function
    # sym=False since we're going for spectral analysis
    hamming_window = hamming(SAMPLES_PER_WINDOW, sym=False)
    print(hamming_window)
    # plt.plot(hamming_window)
    # plt.ylabel("Hamming Window baby!")
    # plt.show()

    windowed = decimated_data[:1024] * hamming_window
    print(windowed)
    # plt.plot(audiodata[:1024])
    # plt.plot(windowed)
    # plt.show()

    fft_data= fft(windowed)
    plt.plot(fft_data)
    plt.show()
    # plt.plot(fft_demo)
    # plt.show()

    # plt.plot(windowed)
    # plt.plot(filtered)
    # test_wav = wave.open('example.wav')
    # test_wav.setnchannels(1)
    # test_wav.setsampwidth(2)
    # test_wav.setframerate(11025)
    # print(test_wav.getnchannels(), test_wav.getframerate())
