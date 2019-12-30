import scipy.io.wavfile as wavfile
from scipy.signal import decimate, butter, filtfilt, spectrogram
from scipy.signal.windows import hamming
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from skimage import util  # operate on all windows in one line
from pydub import AudioSegment  # convert mp3 to wav
import subprocess
import os
from shutil import copyfile

import pyaudio  # To record and playback audio.
# NOTE: Linux users should install PortAudio from their respective distro repos

import wave  # To writeback the recorded samples as .wav files.

# Not using scipy.wavfile.write because it doesn't provide finer controls


DEFAULT_SAMPLING_RATE = 44100
SAMPLING_RATE = 11025
CUTOFF_FREQUENCY = 5000
SAMPLES_PER_WINDOW = 4096
TIME_RESOLUTION = SAMPLES_PER_WINDOW / SAMPLING_RATE
UPPER_FREQ_LIMIT = 600
# Frequency ranges for hashing
RANGES = [40, 80, 120, 180, UPPER_FREQ_LIMIT + 1]


def read_audio_file(filename):
    file_name, file_extension = os.path.splitext(filename)
    if file_extension != '.wav':
        filename = convert_to_wav(filename)
    else:
        copyfile(filename, os.path.join('Songs', os.path.basename(filename)))
    rate, data = wavfile.read(filename)
    return rate, data


def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2


def convert_to_wav(filename):
    """
    Converts any audio format to wav supported by FFMPEG
    """
    try:
        source_parent = os.path.dirname(filename)
        filename = os.path.basename(filename)
        # song_title = filename.split('.')[0]
        # song_format = filename.split('.')[1]
        song_title, song_format = os.path.splitext(filename)
        exported_song_title = song_title + '.wav'
        original_song = AudioSegment.from_file(
            os.path.join(source_parent, filename), format=song_format[1:])
        original_song = original_song.set_channels(1)
        original_song = original_song.set_frame_rate(44100)
        exported_song = original_song.export(os.path.join(
            'Songs', exported_song_title), format="wav")
        return os.path.join(
            'Songs', exported_song_title)
    except IndexError:
        return None


def butter_lowpass(cutoff, fs, order=5):
    """
    :param cutoff:
    :param fs:
    :param order:
    :return: b, a - Filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :return: Filtered data
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def downsample_signal(data, factor):
    return decimate(data, factor)


def apply_window_function(data, window_size, window_function):
    # (window_size,) ==> dimensions are window_size x 1
    windows = util.view_as_windows(data, window_shape=(window_size,), step=100)
    windows = windows * window_function
    return windows


def fft_demo(data, window_size, window_function):
    fft_data = fft(data[:window_size] * window_function)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data[:window_size // 2]), freq


def fft_one_window(window, window_size):
    """
    FFT on a single window
    :param window:
    :param window_size:
    :return: FFT of the window
    """
    fft_data = fft(window)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data)[:window_size // 2], freq[:window_size // 2]


def plot_spectrogram(data, window_size, sampling_rate):
    """
    Plot unfiltered spectrogram
    :param data:
    :param window_size:
    :param sampling_rate:
    """
    freq, time, Spectrogram = spectrogram(data, fs=sampling_rate,
                                          window='hamming', nperseg=window_size,
                                          noverlap=window_size - 100, detrend=False,
                                          scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(time, freq / 1000, np.log10(Spectrogram), cmap="PuOr")
    ax.set_ylabel('Frequency [kHz]', fontsize=22)
    ax.set_xlabel('Time [sec]', fontsize=22)
    plt.title("Renai Circulation (Bakemonogatari OP)", fontsize=22)
    plt.show()


def filter_spectrogram(windows, window_size):
    """
    Filters spectrogram data
    :param windows:
    :param window_size:
    :return: filtered_bins
    """
    # Init 2D list
    filtered_bins = [[0 for i in range(len(RANGES))] for j in range(
        len(windows))]  # rows = no. of windows, cols = no. of bands

    for i in range(len(windows)):
        fft_data, freq = fft_one_window(windows[i], window_size)
        max_amp_freq_value = 0
        max_amp = 0
        current_freq_range_index = 0
        for j in range(len(fft_data)):

            if freq[j] > UPPER_FREQ_LIMIT:
                continue

            # Reset max. amplitudes and bins for each band
            if current_freq_range_index != return_freq_range_index(freq[j]):
                current_freq_range_index = return_freq_range_index(freq[j])
                max_amp_freq_value = 0
                max_amp = 0

            if fft_data[j] > max_amp:
                max_amp = fft_data[j]
                max_amp_freq_value = freq[j]

            filtered_bins[i][current_freq_range_index] = max_amp_freq_value

    return filtered_bins


def return_freq_range_index(freq_value):
    """
    Returns band index for a given freq_value
    :param freq_value:
    :return: freq_range_index
    """
    freq_range_index = 0
    while freq_value > RANGES[freq_range_index]:
        freq_range_index = freq_range_index + 1
    return freq_range_index


def plot_filtered_spectrogram(filtered_data):
    """
    Plot filtered spectrogram
    :param filtered_data:
    """
    for window_index in range(len(filtered_data)):
        """
        The function np.array generates a numpy array
        Here, np.array(1*3) => [3]
              np.array([1]*3) => [1,1,1]
        This is why, in the code below, the term 'window_index' is inside square brackets.
        All in all, it generates an array of size equal to no. of bands with all values equal to
        window_index*time_resolution (to convert window indices to time values)
        """
        timestamp = np.array(
            [window_index] * len(filtered_data[window_index])) * TIME_RESOLUTION

        # Scatter plot of filtered bins
        # c => color of point, marker => shape of mark
        plt.scatter(timestamp, filtered_data[window_index], c='b', marker='.')

    # To force the graph to be plotted upto 512 even though our y values range
    # from 0 to 300
    plt.ylim(0, 512)

    # Below loop draws horizontal lines for each band
    for i in range(len(RANGES)):
        plt.axhline(y=RANGES[i], c='r')
    plt.show()


def song_recipe(filename):
    """
    Run the entire algorithm on a particular song
    :param filename:
    :return filtered_spectrogram_data:
    """
    rate, audio_data = read_audio_file(filename)
    if audio_data.ndim != 1:  # Checks no. of channels. Some samples are already mono
        audio_data = stereo_to_mono(audio_data)
    filtered_data = butter_lowpass_filter(
        audio_data, CUTOFF_FREQUENCY, DEFAULT_SAMPLING_RATE)
    decimated_data = downsample_signal(
        filtered_data, DEFAULT_SAMPLING_RATE // SAMPLING_RATE)
    hamming_window = hamming(SAMPLES_PER_WINDOW, sym=False)
    windows = apply_window_function(
        decimated_data, SAMPLES_PER_WINDOW, hamming_window)
    filtered_spectrogram_data = filter_spectrogram(windows, SAMPLES_PER_WINDOW)
    return filtered_spectrogram_data
