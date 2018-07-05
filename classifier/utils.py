import logging
import random
from itertools import chain
from math import ceil
from typing import List, Callable, Any

import numpy as np
from gensim.models import Word2Vec
from more_itertools import chunked
from pathos.multiprocessing import Pool, cpu_count

NUMBER_OF_SEEDS = 10


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    songs_and_tracks = np.load(filename)
    logging.getLogger('logging_songscuncert').debug('number of loaded playists:'+str(len(songs_and_tracks)))
    return songs_and_tracks


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def process_play_list_constructor(target_embeddings_file, context_embeddings_file):
    """Generate a function that will clean and tokenize text."""
    def process_play_list(playlist_and_tracks):
        samples = []
        target_embeddings = load_embeddings(target_embeddings_file)
        context_embeddings = load_embeddings(context_embeddings_file)
        try:
            for play_list_songs in playlist_and_tracks:
                playlist, songs = zip(play_list_songs)
                playlist = playlist[0]
                songs = songs[0]
                found, target_index = get_valid_song(context_embeddings, songs)
                # if we got a valid target continue
                if found:
                    tries = 0
                    count = 0
                    seeds = None
                    while count < NUMBER_OF_SEEDS and tries < 100:
                        found_seed, seed_index = get_valid_song(context_embeddings, songs, target_index)
                        if found_seed:
                            count += 1
                            if seeds is None:
                                seeds = context_embeddings.wv.vectors[int(seed_index)]
                                tries = 0
                            else:
                                seeds = np.hstack((seeds, context_embeddings.wv.vectors[int(seed_index)]))
                        else:
                            tries += 1
                    # if there valid seeds to calculate the average
                    if count >= NUMBER_OF_SEEDS:
                        # get the embedding for the playlist
                        playlist_embedding = target_embeddings.wv.vectors[int(playlist)]
                        # get the embedding of the target
                        target_embedding = context_embeddings.wv.vectors[int(songs[target_index])]
                        found_negative, negative_sample_index = get_valid_neg_sample(context_embeddings, songs)
                        #if we find a valid neg sample proceed to build the samples
                        if found_negative:
                            # add pos sample together with the corresponding ids
                            samples.append((playlist, songs[target_index], playlist_embedding,
                                            seeds, target_embedding, 1))
                            # get the neg sample embedding
                            negative_sample_embedding = \
                                context_embeddings.wv.vectors[negative_sample_index]
                            # add the neg sample together with the corresponding ids
                            samples.append((playlist, str(negative_sample_index), playlist_embedding,
                                            seeds, negative_sample_embedding, 0))
        except Exception as e:
            logging.getLogger('logging_songscuncert').error('error '+e)
        return samples

    def get_valid_song(context_embeddings, songs, target_index = None):
        found = False
        count = 0
        # try 10 times to get a random seed that has a valid embedding
        while not found and count < 10:
            song_index = random.randint(0, len(songs) - 1)
            if songs[song_index] in context_embeddings.wv.vocab:
                if target_index is None or songs[song_index] != target_index:
                    found = True
            else:
                count += 1
        return found, song_index

    def get_valid_neg_sample(context_embeddings, songs):
        found_negative = False
        counter = 0
        # try 100 times to find a valid neg sample picked randomly
        while not found_negative and counter < 100:
            negative_sample_index = random.randint(0, len(context_embeddings.wv.vectors) - 1)
            if str(negative_sample_index) in context_embeddings.wv.vocab \
                    and str(negative_sample_index) not in songs:
                found_negative = True
            else:
                counter += 1
        return found_negative, negative_sample_index

    return process_play_list


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    """
    Apply function to list of elements.

    Automatically determines the chunk size.
    """
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data


def load_embeddings(emb_file):
    w2v_model = Word2Vec.load(emb_file)
    return w2v_model

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create logger
    logger = logging.getLogger("logging_songscuncert")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


def batch_generator(n_minibatch, input_samples, input_playlists, input_seeds, input_labels):
    batch_size = n_minibatch
    data_samples = input_samples
    data_playlists = input_playlists
    data_seeds = input_seeds
    data_labels = input_labels
    while True:
        if data_samples.shape[0] < batch_size:
            data_samples = np.concatenate([data_samples, input_samples])
            data_playlists = np.concatenate([data_playlists, input_playlists])
            data_seeds = np.concatenate([data_seeds, input_seeds])
            data_labels = np.concatenate([data_labels, input_labels])
            if data_samples.shape[0] < batch_size:
                continue
        samples = data_samples[:batch_size]
        playlists = data_playlists[:batch_size]
        seeds = data_seeds[:batch_size]
        labels = data_labels[:batch_size]
        data_samples = data_samples[batch_size:]
        data_playlists = data_playlists[batch_size:]
        data_seeds = data_seeds[batch_size:]
        data_labels = data_labels[batch_size:]
        yield samples, playlists, seeds, labels


def feed(batch, samples_placeholder, playlists_placeholder, seeds_placeholder, labels_placeholder):
    samples, playlists, seeds, labels = batch.__next__()
    return {samples_placeholder: samples,
            playlists_placeholder: playlists,
            seeds_placeholder: seeds,
            labels_placeholder: labels}