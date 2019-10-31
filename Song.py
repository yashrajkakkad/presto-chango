import scipy.io.wavfile as wavfile
from scipy.signal import decimate, butter, filtfilt, spectrogram
from scipy.signal.windows import hamming
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from skimage import util  # operate on all windows in one line
from pydub import AudioSegment  # convert mp3 to wav

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
    exported_song = song_title + '.wav'
    AudioSegment.from_file(filename, format=song_format).export(exported_song, format="wav")
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
    downsampled_data = decimate(data, factor)
    return downsampled_data


def apply_window_function(data, window_size, window_function):
    windows = util.view_as_windows(data, window_shape=(window_size,),
                                   step=100)  # (window_size,) ==> dimensions are window_size x 1
    windows = windows * window_function
    return windows


def fft_demo(data, window_size, window_function, sampling_rate):
    # fft_data = fft(data[:window_size]*window_function)
    # fft_data = np.multiply(fft(data[:window_size]), window_function)
    # # plt.plot(fft_data)
    # fft_freq = np.fft.fftfreq(window_size//2)
    # power = np.abs(fft_data[:window_size//2])
    # plt.subplot(2, 1, 1)
    # # plt.plot(, power)
    # plt.plot(np.abs(fft_freq)*sampling_rate, np.abs(fft_data)[:window_size//2])
    # plt.subplot(2, 1, 2)
    # plt.plot(power)
    # plt.show()
    fft_data = fft(data[:window_size] * window_function)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data[:window_size // 2]), freq


# FFT on a single window
def fft_one_window(window):
    fft_data = fft(window)
    freq = fftfreq(len(fft_data), 1 / SAMPLING_RATE)
    return np.abs(fft_data), freq


# Plot unfiltered spectrogram
def plot_spectrogram(data, window_size, sampling_rate):
    freq, time, spectrogram = spectrogram(data, fs=sampling_rate, window='hamming', nperseg=window_size,
                                          noverlap=window_size - 100, detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(time, freq / 1000, np.log10(spectrogram), cmap="PuOr")
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
    filtered_bins = [[0 for i in range(len(RANGES))] for j in
                     range(len(windows))]  # rows = no. of windows, cols = no. of bands

    for i in range(len(windows)):
        fft_data, freq = fft_one_window(windows[i][:window_size // 2])
        max_amp_bin = 0
        max_amp = 0
        current_band_index = 0
        for j in range(len(fft_data)):

            if j > UPPER_FREQ_LIMIT:
                continue

            # Reset max. amplitudes and bins for each band
            if current_band_index != return_band_index(j):
                current_band_index = return_band_index(j)
                max_amp_bin = 0
                max_amp = 0

            if fft_data[j] > max_amp:
                max_amp = fft_data[j]
                max_amp_bin = j

            filtered_bins[i][current_band_index] = max_amp_bin
    return filtered_bins


# Returns band index for a givne bin_index
def return_band_index(bin_index):
    band_index = 0;
    while bin_index > RANGES[band_index]:
        band_index = band_index + 1
    return band_index


# Plot filtered spectrogram
def plot_filtered_spectrogram(filtered_data):
    for window_index in range(len(filtered_data)):
        """
        The function np.array generates a numpy array
        Here, np.array(1*3) => [3]
              np.array([1]*3) => [1,1,1]
        This is why, in the code below, the term 'window_index' is inside square brackets.
        All in all, it generates an array of size equal to no. of bands with all values equal to window_index*time_resolution (to convert window indices to time values)
        """
        timestamp = np.array([window_index] * len(filtered_data[window_index])) * TIME_RESOLUTION

        # Scatter plot of filtered bins
        # c => color of point, marker => shape of mark
        plt.scatter(timestamp, filtered_data[window_index], c='b', marker='.')

    plt.ylim(0, 512)  # To force the graph to be plotted upto 512 even though our y values range from 0 to 300

    # Below loop draws horizontal lines for each band
    for i in range(len(RANGES)):
        plt.axhline(y=RANGES[i], c='r')
    plt.show()


if __name__ == "__main__":

    # Read the audio file
    filename = 'AudioSamples/modem.wav'
    rate, audio_data = read_audio_file(filename)
    assert (rate == DEFAULT_SAMPLING_RATE)  # Some samples are sampled at 48 kHz

    # Convert stereo to mono
    if audio_data.ndim != 1:  # Checks no. of channels. Some samples are already mono
        audio_data = stereo_to_mono(audio_data)
    # print(audio_data)

    # Pass the signal to a low-pass filter with a cutoff frequency of 5000 Hz
    filtered_data = butter_lowpass_filter(audio_data, CUTOFF_FREQUENCY, DEFAULT_SAMPLING_RATE)

    # Decimate by a factor of 4
    decimated_data = downsample_signal(filtered_data, DEFAULT_SAMPLING_RATE // SAMPLING_RATE)
    # print(DEFAULT_SAMPLING_RATE // SAMPLING_RATE)
    # print(decimated_data)
    first_len = len(filtered_data)
    second_len = len(decimated_data)
    # print(first_len, second_len)
    # assert (first_len / 4 == second_len)

    # plot_spectrogram(decimated_data, SAMPLES_PER_WINDOW, SAMPLING_RATE)

    # Generate a hamming window function
    # sym=False since we're going for spectral analysis
    hamming_window = hamming(SAMPLES_PER_WINDOW, sym=False)
    # print(hamming_window)
    # plt.plot(hamming_window)
    # plt.ylabel("Hamming Window baby!")
    # plt.show()

    windows = apply_window_function(decimated_data, SAMPLES_PER_WINDOW, hamming_window)

    # FFT on a single window
    # fft_data, freq = fft_demo(decimated_data, SAMPLES_PER_WINDOW, hamming_window, SAMPLING_RATE)
    # plt.plot(freq, np.abs(fft_data))
    # plt.xlabel("Frequency [Hz]", fontsize=22)
    # plt.ylabel("|X(w)|", fontsize=22)
    # plt.show()

    # TODO: Under review
    filtered_spectrogram_data = filter_spectrogram(windows, SAMPLES_PER_WINDOW)

    plot_filtered_spectrogram(filtered_spectrogram_data)
