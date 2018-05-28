import random

import tensorflow as tf
import numpy as np
from pathos.multiprocessing import Pool, cpu_count
from more_itertools import chunked
from typing import List, Callable, Union, Any
from math import ceil
from itertools import chain
import logging


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    songs_and_tracks = np.load(filename)
    logging.debug(f'loaded {len(songs_and_tracks)} playists')
    return songs_and_tracks[:10000]


def flattenlist(listoflists):
    return list(chain.from_iterable(listoflists))


def process_play_list_constructor(neg_samples:int, dictionary:dict):
    """Generate a function that will clean and tokenize text."""
    def process_play_list(play_lists):
        samples = []
        dictionary_keys = list(dictionary.keys())
        try:
            for play_list in play_lists:
                for song in play_list[1]:
                    if song not in dictionary:
                        song = 'UNK'
                    samples.append((int(play_list[0]), dictionary[song], 1))
                    for i in range(neg_samples):
                        random_neg_sample = random.randint(0, len(dictionary) - 1)
                        samples.append((int(play_list[0]), dictionary[dictionary_keys[random_neg_sample]], 0))
        except Exception as e:
            logging.error(f'error {e}')
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

