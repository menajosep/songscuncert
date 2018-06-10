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
    return songs_and_tracks


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def get_optimal():
    x = np.linspace(0, 1, 100)
    a = 1
    for b in range(7, 17, 2):
        optimal = np.power(x, a) * np.power((1 - x), b) + 1e-3
        optimal = optimal / np.sum(optimal)
    return optimal


def is_goog_embedding(sigmas):
    threshold = 0.5
    optimal = get_optimal()
    hist = plt.hist(sigmas, bins=100, color='green', label='sigma values')
    distr = (hist[0] + 1e-5) / np.sum(hist[0])
    distance = -np.sum(optimal * np.log(distr / optimal))
    return distance < threshold


def process_play_list_constructor(neg_samples:int, dictionary:dict, context_size:int, sampling_table:dict):
    """Generate a function that will clean and tokenize text."""
    def process_play_list(play_lists):
        samples = []
        dictionary_keys = list(dictionary.keys())
        try:
            for play_list in play_lists:
                songs = play_list[1]
                shuffle(songs)
                for song in songs[:context_size]:
                    if song in dictionary and sampling_table[song] < random.random():
                        songs2 = songs.copy()
                        shuffle(songs2)
                        for song2 in songs2[:context_size]:
                            if song2 not in dictionary:
                                song2 = 'UNK'
                            if song != song2:
                                samples.append((dictionary[song], dictionary[song2], 1))

                        for i in range(neg_samples):
                            random_neg_sample = random.randint(0, len(dictionary) - 1)
                            samples.append((dictionary[song], dictionary[dictionary_keys[random_neg_sample]], 0))
        except Exception as e:
            logging.getLogger('logging_songscuncert').error('error '+str(e))
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

