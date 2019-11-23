import pyaudio
import wave
from song import song_recipe
from database import hash_sample, load_database, find_song

SAMPLE_DURATION = 30


def record_sample_recipe(filename="sample.wav"):
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
    print("Recording started")
    record_sample_recipe()
    print("Recording finished")
    filtered_bins_sample = song_recipe("sample.wav")
    sample_dict = hash_sample(filtered_bins_sample)
    song_to_id, id_to_song, hash_dict = load_database()
    offset_dict, song_id = find_song(
        hash_dict, sample_dict, id_to_song)
    print(song_id)
    print(id_to_song)
    print(id_to_song[song_id])
    playback_recorded_sample()
