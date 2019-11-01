import scipy.io.wavfile as wavfile
from scipy.signal import decimate, butter, filtfilt, spectrogram
from scipy.signal.windows import hamming
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from skimage import util  # operate on all windows in one line
from pydub import AudioSegment  # convert mp3 to wav

import pyaudio  # To record and playback audio.
# NOTE: Linux users should install PortAudio from their respective distro repos

import wave  # To writeback the recorded samples as .wav files.
# Not using scipy.wavfile.write because it doesn't provide finer controls


DEFAULT_SAMPLING_RATE = 44100
SAMPLING_RATE = 11025
CUTOFF_FREQUENCY = 5000
SAMPLES_PER_WINDOW = 1024
TIME_RESOLUTION = SAMPLES_PER_WINDOW / SAMPLING_RATE
UPPER_FREQ_LIMIT = 300
RANGES = [40, 80, 120, 180, UPPER_FREQ_LIMIT + 1]


def read_audio_file(filename):
    if filename.split('.')[1] == 'wav':
        rate, data = wavfile.read(filename)
        return rate, data
    else:
        converted_file = convert_to_wav(filename)
        rate, data = wavfile.read(converted_file)
        return rate, data


def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2


def convert_to_wav(filename):
    song_title = filename.split('.')[0]
    song_format = filename.split('.')[1]
    # song_title, song_format = filename.split('.')[0:2]
    exported_song = song_title + '.wav'
    AudioSegment.from_file(filename, format=song_format).export(
        exported_song, format="wav")
    return exported_song


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def downsample_signal(data, factor):
    # downsampled_data = decimate(data, factor)
    # return downsampled_data
    return decimate(data, factor)


def apply_window_function(data, window_size, window_function):
    # (window_size,) ==> dimensions are window_size x 1
    windows = util.view_as_windows(data, window_shape=(window_size,), step=100)
    windows = windows * window_function
    return windows


def fft_demo(data, window_size, window_function):
    # fft_data = fft(data[:window_size]*window_function)
    # fft_data = np.multiply(fft(data[:window_size]), window_function)
    # # plt.plot(fft_data)
    # fft_freq = np.fft.fftfreq(window_size//2)
    # power = np.abs(fft_data[:window_size//2])
    # plt.subplot(2, 1, 1)
    # # plt.plot(, power)
    # plt.plot(np.abs(fft_freq)*sampling_rate,
    #          np.abs(fft_data)[:window_size//2])
    # plt.subplot(2, 1, 2)
    # plt.plot(power)
    # plt.show()
    fft_data = fft(data[:window_size] * window_function)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data[:window_size // 2]), freq


# FFT on a single window
def fft_one_window(window, window_size):
    fft_data = fft(window)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data)[:window_size // 2], freq[:window_size // 2]


# Plot unfiltered spectrogram
def plot_spectrogram(data, window_size, sampling_rate):
    freq, time, Spectrogram = spectrogram(data, fs=sampling_rate,
                                          window='hamming', nperseg=window_size,
                                          noverlap=window_size-100, detrend=False,
                                          scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(time, freq / 1000, np.log10(Spectrogram), cmap="PuOr")
    ax.set_ylabel('Frequency [kHz]', fontsize=22)
    ax.set_xlabel('Time [sec]', fontsize=22)
    plt.title("Renai Circulation (Bakemonogatari OP)", fontsize=22)
    plt.show()


# Filter spectrogram data
def filter_spectrogram(windows, window_size):
    # OLD CODE
    # spectrum = np.fft.fft(windows, axis=0)[:window_size // 2 + 1:-1]
    # spectrum = np.abs(spectrum)
    # freqs = np.fft.fftfreq(window_size // 2)
    # plt.plot(freqs, spectrum)

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


# Returns band index for a given freq_value
def return_freq_range_index(freq_value):
    freq_range_index = 0
    while freq_value > RANGES[freq_range_index]:
        freq_range_index = freq_range_index + 1
    return freq_range_index


# Plot filtered spectrogram
def plot_filtered_spectrogram(filtered_data):
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


def record_sample_recipe(filename, duration):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 30
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def playback_recorded_sample(filename):
    chunk = 1024  # Set chunk size of 1024 samples per data frame

    wf = wave.open(filename, 'rb')  # Open the sound file

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)  # Read data in chunks

    # Play the sound by writing the audio data to the stream
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

    # Close and terminate the stream
    stream.close()
    p.terminate()


def song_recipe(filename):
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
    # plot_filtered_spectrogram(filtered_spectrogram_data)


if __name__ == "__main__":

    # Read the audio file
    filename = 'Songs/RenaiCirculation.wav'

    filtered_spectrogram_data = song_recipe(filename)
    plot_filtered_spectrogram(filtered_spectrogram_data)

    # record_sample_recipe('output.wav',30)
    # filtered_spectrogram_data = song_recipe('output.wav')
    # plot_filtered_spectrogram(filtered_spectrogram_data)

    # playback_recorded_sample('output.wav')

    # Convert stereo to mono, if required
    # if audio_data.ndim != 1:  # Checks no. of channels. Some samples are already mono
    #     audio_data = stereo_to_mono(audio_data)
    # # print(audio_data)

    # Pass the signal to a low-pass filter with a cutoff frequency of 5000 Hz
    # filtered_data = butter_lowpass_filter(audio_data, CUTOFF_FREQUENCY, DEFAULT_SAMPLING_RATE)

    # Decimate by a factor of 4
    # decimated_data = downsample_signal(filtered_data, DEFAULT_SAMPLING_RATE // SAMPLING_RATE)
    # print(DEFAULT_SAMPLING_RATE // SAMPLING_RATE)
    # print(decimated_data)
    # first_len = len(filtered_data)
    # second_len = len(decimated_data)
    # print(first_len, second_len)
    # assert (first_len / 4 == second_len)

    # plot_spectrogram(decimated_data, SAMPLES_PER_WINDOW, SAMPLING_RATE)

    # Generate a hamming window function
    # sym=False since we're going for spectral analysis
    # hamming_window = hamming(SAMPLES_PER_WINDOW, sym=False)
    # print(hamming_window)
    # plt.plot(hamming_window)
    # plt.ylabel("Hamming Window baby!")
    # plt.show()

    # windows = apply_window_function(decimated_data, SAMPLES_PER_WINDOW, hamming_window)
    # print(type(windows))
    # print(windows)
    # print(len(decimated_data))
    # print(len(windows))

    # FFT on a single window
    # fft_data, freq = fft_demo(decimated_data, SAMPLES_PER_WINDOW, hamming_window, SAMPLING_RATE)
    # plt.plot(freq, np.abs(fft_data))
    # plt.xlabel("Frequency [Hz]", fontsize=22)
    # plt.ylabel("|X(w)|", fontsize=22)
    # plt.show()

    # TODO: Under review
    # filtered_spectrogram_data = filter_spectrogram(windows, SAMPLES_PER_WINDOW)
