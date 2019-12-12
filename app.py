import pyaudio
import wave
from song import song_recipe, plot_filtered_spectrogram
from database import hash_sample, load_database, find_song, create_database
import operator

SAMPLE_DURATION = 30


def record_sample_recipe(filename="sample.wav"):
    """
    Record a song
    :param filename:
    """
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = SAMPLE_DURATION

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


def playback_recorded_sample(filename="sample.wav"):
    """
    Playback recorded song
    :param filename:
    """
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


if __name__ == "__main__":
    # create_database()
    print("Music identification with audio fingerprinting")
    print("Loading database")
    print(".\n.\n.")
    song_to_id, id_to_song, hash_dict = load_database()
    print("Database loaded")
    print("\nWelcome!")
    while True:
        # print("Recording started")
        # record_sample_recipe()
        # print("Recording finished")
        print("Enter file name: ")
        filename = input()
        filtered_bins_sample = song_recipe(filename)  # Run our algorithm on the song
        # plot_filtered_spectrogram(filtered_bins_sample)

        sample_dict = hash_sample(filtered_bins_sample)
        max_frequencies, max_frequencies_keys = find_song(
            hash_dict, sample_dict, id_to_song)
        count = 0
        for song_id in max_frequencies_keys:
            print(id_to_song[song_id], max_frequencies[song_id])
            count += 1
            if count == 5:
                break
        # print(song_id)
        # print(id_to_song)
        # print(offset_dict)
        # print(id_to_song[song_id])
        # print(sorted(offset_dict.items(), key=operator.itemgetter(1)))
        # playback_recorded_sample()
        print("Would you like to test another? 1/0")
        choice = int(input())
        if not choice:
            break
