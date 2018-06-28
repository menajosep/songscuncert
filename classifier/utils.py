import logging
import random
from itertools import chain
from math import ceil
from typing import List, Callable, Any

import numpy as np
from gensim.models import Word2Vec
from more_itertools import chunked
from pathos.multiprocessing import Pool, cpu_count


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
                found, target_index = get_valid_target(context_embeddings, songs)
                # if we got a valid target continue
                if found:
                    count = 0
                    average = np.zeros(len(context_embeddings.wv.vectors[0]))
                    seeds = None
                    # get those seeds that have a valid embedding
                    for song in songs:
                        if song in context_embeddings.wv.vocab and song != songs[target_index]:
                            if count < 3:
                                if seeds is None:
                                    seeds = context_embeddings.wv.vectors[int(song)]
                                else:
                                    seeds = np.hstack((average, context_embeddings.wv.vectors[int(song)]))
                            average = np.add(average, context_embeddings.wv.vectors[int(song)])
                            count += 1
                    # if there valid seeds to calculate the average
                    if count > 0:
                        # get the embedding for the playlist
                        playlist_embedding = target_embeddings.wv.vectors[int(playlist)]
                        # get the centroid as the average of the embeddings of the valid seeds
                        average = average / count
                        # get the top similar songs for the centroid
                        top = context_embeddings.similar_by_vector(average, topn=1000, restrict_vocab=None)
                        # get the embedding of the targe
                        target_embedding = context_embeddings.wv.vectors[int(songs[target_index])]
                        found_negative, negative_sample_index = get_valid_neg_sample(context_embeddings, songs, top)
                        #if we find a valid neg sample proceed to build the samples
                        if found_negative:
                            # get the neg sample embedding
                            negative_sample_embedding = context_embeddings.wv.vectors[int(top[negative_sample_index][0])]
                            # build the pos sample
                            target_sample = np.hstack((playlist_embedding, seeds, target_embedding))
                            # add it together with the corresponding ids
                            samples.append((playlist, songs[target_index], target_sample, 1))
                            # build the neg sample
                            negative_sample_sample = np.hstack((playlist_embedding, seeds, negative_sample_embedding))
                            # add it together with the corresponding ids
                            samples.append((playlist, top[negative_sample_index][0], negative_sample_sample, 0))
        except Exception as e:
            logging.getLogger('logging_songscuncert').error('error '+e)
        return samples



    def get_valid_target(context_embeddings, songs):
        found = False
        count = 0
        # try 10 times to get a random seed that has a valid embedding
        while not found and count < 10:
            target_index = random.randint(0, len(songs) - 1)
            if songs[target_index] in context_embeddings.wv.vocab:
                found = True
            else:
                count += 1
        return found, target_index

    def get_valid_neg_sample(context_embeddings, songs, top):
        found_negative = False
        counter = 0
        # try 100 times to find a valid neg sample picked randomly from the top similar
        # songs to the centroid
        while not found_negative and counter < 100:
            negative_sample_index = random.randint(0, len(top) - 1)
            if top[negative_sample_index][0] in context_embeddings.wv.vocab \
                    and top[negative_sample_index][0] not in songs:
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


def batch_generator(n_minibatch, input_samples, input_labels):
    batch_size = n_minibatch
    data_samples = input_samples
    data_labels = input_labels
    while True:
        if data_samples.shape[0] < batch_size:
            data_samples = np.concatenate([data_samples, input_samples])
            data_labels = np.concatenate([data_labels, input_labels])
            if data_samples.shape[0] < batch_size:
                continue
        samples = data_samples[:batch_size]
        labels = data_labels[:batch_size]
        data_samples = data_samples[batch_size:]
        data_labels = data_labels[batch_size:]
        yield samples, labels


def feed(batch, samples_placeholder, labels_placeholder, shuffling = False):
    samples, labels = batch.__next__()
    if shuffling:
        labels = np.random.permutation(labels)
    return {samples_placeholder: samples,
            labels_placeholder: labels}