import random

import tensorflow as tf
import numpy as np
from pathos.multiprocessing import Pool, cpu_count
from more_itertools import chunked
from typing import List, Callable, Union, Any
from math import ceil
from itertools import chain
import logging
from random import shuffle
import matplotlib.pyplot as plt


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    songs_and_tracks = np.load(filename)
    logging.getLogger('logging_songscuncert').debug('number of loaded playists:'+str(len(songs_and_tracks)))
    return songs_and_tracks[:6000]


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def get_optimal():
    x = np.linspace(0, 1, 100)
    a = 1
    optimals = []
    for b in range(7, 17, 2):
        optimal = np.power(x, a) * np.power((1 - x), b) + 1e-3
        optimal = optimal / np.sum(optimal)
        optimals.append(optimal)
    return optimals


def is_goog_embedding(sigmas):
    threshold = 1e-3
    optimals = get_optimal()
    hist = plt.hist(sigmas, bins=100, color='green', label='sigma values')
    distr = (hist[0] + 1e-5) / np.sum(hist[0])
    distance = 0
    for optimal in optimals:
        distance += -np.sum(optimal * np.log(distr / optimal))
    distance = distance / len(optimals)
    return distance < threshold


def process_play_list_constructor(context_embeddings, target_embeddings, playlist_and_tracks):
    """Generate a function that will clean and tokenize text."""
    def process_play_list(play_lists):
        samples = []
        try:
            for play_list_id in play_lists:
                playlist, songs = zip(playlist_and_tracks[play_list_id])
                playlist_embedding = target_embeddings.wv.vectors[int(playlist[0])]
                count = 0
                average = np.zeros(len(context_embeddings.wv.vectors[0]))
                for song in songs[0]:
                    if song in context_embeddings.wv.vocab:
                        average = np.add(average, context_embeddings.wv.vectors[int(song)])
                        count += 1
                average = average / count
                top2000 = context_embeddings.similar_by_vector(average, topn=2000, restrict_vocab=None)
                target_index = random.randint(0, count - 1)
                target_embedding = context_embeddings.wv.vectors[target_index]
                negative_sample_index = random.randint(0, len(top2000) - 1)
                negative_sample_embedding = context_embeddings.wv.vectors[negative_sample_index]
                target_sample = np.hstack((playlist_embedding, average, target_embedding))
                samples.append((target_sample, 1))
                negative_sample_sample = np.hstack((playlist_embedding, average, negative_sample_embedding))
                samples.append((negative_sample_sample, 0))



        except Exception as e:
            logging.getLogger('logging_songscuncert').error('error '+e)
        return samples

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


def variable_summaries(summary_name, var):
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


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

