import os
from random import random
from Song import song_recipe


# class DataPoint:
#     def __init__(self, time, song_id):
#         self.time = time
#         self.song_id = song_id

def hash_window(filtered_bin):
    """
    :param filtered_bin: A filtered bin of a window generated by filter_spectrogram
    :return: hash value of the particular bin
    """

    """
    Note that we must assume that the recording is not done in perfect conditions (i.e., a “deaf room”),
    and as a result we must include a fuzz factor.
    Fuzz factor analysis should be taken seriously, and in a real system,
    the program should have an option to set this parameter based on the conditions of the recording.
    """
    fuz_factor = 2  # for error correction TODO: figure out why?

    return (filtered_bin[3] - (filtered_bin[3] % fuz_factor)) * 1e8 + (
            filtered_bin[2] - (filtered_bin[2] % fuz_factor)) * 1e5 + (
                   filtered_bin[1] - (filtered_bin[1] % fuz_factor)) * 1e2 + (
                   filtered_bin[0] - (filtered_bin[0] % fuz_factor))


def hash_song(song_id, filtered_bins, hash_dictionary):
    for filtered_bin in filtered_bins:
        try:
            hash_dictionary[hash_window(filtered_bin)].append(song_id)
        except Exception:
            hash_dictionary[hash_window(filtered_bin)] = [song_id]


def create_database():
    song_to_id = {}
    hash_dictionary = {}
    for dirpath, dirname, filenames in os.walk('Songs'):
        for filename in filenames:
            print(filename)
            song_id = int(random() * 1000)
            song_to_id[filename] = song_id
            filtered_bins = song_recipe("Songs/" + filename)
            hash_song(song_id, filtered_bins, hash_dictionary)
    return song_to_id, hash_dictionary


if __name__ == "__main__":
    song_to_id, hash_dictionary = create_database()
    print(song_to_id)
    print(hash_dictionary)
