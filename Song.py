import wave
import scipy.io.wavfile as wavfile
from scipy.signal import decimate, hamming, butter, lfilter, freqz
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np


# from pydub import AudioSegment
# from pydub.playback import play

def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2


def butter_lowpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=10):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class Song:

    def __init__(self, fileName, windowSize, samplingFrequency):
        self.name = fileName
        self.wSize = windowSize
        self.fs = samplingFrequency


if __name__ == "__main__":
    # test_wav = wave.open('example.wav')
    # test_wav.setnchannels(1)
    # test_wav.setsampwidth(2)
    # test_wav.setframerate(11025)
    # print(test_wav.getnchannels(), test_wav.getframerate())
    rate, audiodata = wavfile.read('example.wav')
    print(rate, audiodata)
    print(type(audiodata))
    # data = data.astype(float)
    audiodata = stereo_to_mono(audiodata)
    print(audiodata)
    first_len = len(audiodata)
    print(len(audiodata))
    audiodata = decimate(audiodata, 4)
    print(audiodata)
    second_len = len(audiodata)
    print(len(audiodata))
    assert (first_len / 4 == second_len)

    # sym=False since we're going for spectral analysis
    hamming_window = hamming(1024, sym=False)
    print(hamming_window)
    # plt.plot(hamming_window)
    # plt.ylabel("Hamming Window baby!")
    # plt.show()

    windowed = audiodata[:1024] * hamming_window
    print(windowed)
    # plt.plot(audiodata[:1024])
    # plt.plot(windowed)
    # plt.show()

    fft_demo = fft(windowed)
    # plt.plot(fft_demo)
    # plt.show()

    filtered = butter_lowpass_filter(windowed, 5000/6, 11025)
    fft_filtered = fft(filtered)
    # plt.plot(windowed)
    # plt.plot(filtered)
    plt.plot(fft_demo)
    plt.plot(fft_filtered)
    plt.show()