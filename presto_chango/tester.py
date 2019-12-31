from presto_chango.song import song_recipe
from presto_chango.database import hash_sample, create_database, load_database, find_song
import scipy.io.wavfile as wavfile
import os
import shlex
import subprocess
from random import randint
import datetime
import youtube_dl

SAMPLE_DURATION = 30


def download_songs(playlist_url):
    """
    Download songs from YouTube, for use in our database
    :param playlist_url:
    """
    command_string = 'youtube-dl -x --audio-format wav --postprocessor-args "-ar 44100 -ac 1" --output "Songs/%(' \
                     'title)s_%(id)s.%(ext)s" ' + \
                     playlist_url
    args = shlex.split(command_string)
    subprocess.call(args)


def gen_random_samples():
    """
    Extract a random 30 second part of a song for testing
    :return:
    """
    if os.path.exists('Song_Samples'):
        pass
    else:
        os.mkdir('Song_Samples')
    for filename in os.listdir("Songs"):
        rate, data = wavfile.read(os.path.join("Songs", filename))
        song_duration = len(data) // rate
        start_point = randint(0, song_duration - SAMPLE_DURATION)
        end_point = start_point + SAMPLE_DURATION
        subprocess.call(['ffmpeg', '-i', os.path.join("Songs", filename),
                         '-ss', str(datetime.timedelta(seconds=start_point)), '-to',
                         str(datetime.timedelta(seconds=end_point)), '-y', os.path.join("Song_Samples", filename)])


def hash_random_sample(filename):
    """
    :param filename:
    :return sample_dict:
    """
    filtered_bins_sample = song_recipe(os.path.join("Song_Samples", filename))
    sample_dict = hash_sample(filtered_bins_sample)
    return sample_dict


def test_accuracy():
    """
    Test the code with random samples
    :return:
    """
    hits = 0
    total = 0
    # create_database()
    # gen_random_samples()
    song_to_id, id_to_song, hash_dict = load_database()
    for filename in os.listdir("Songs"):
        sample_dict = hash_random_sample(filename)
        offset_dict, song_id = find_song(
            hash_dict, sample_dict, id_to_song)
        print(id_to_song[song_id])
        print(filename)
        if id_to_song[song_id] == filename:
            print("Success")
            hits += 1
        else:
            print("Fail")
        total += 1
    print((hits / total) * 100, " %")


"""
download_songs(
"https://www.youtube.com/playlist?
list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj")
"""
