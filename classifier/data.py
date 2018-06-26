from math import sqrt

from gensim.models import Word2Vec

from classifier.utils import *
from random import shuffle
import collections

class classifier_data():
    def __init__(self, input_file, target_emb_file, context_emb_file, dir_name, logger):
        self.logger = logger
        self.logger.debug('initializing classifier_data with file ' + input_file)
        self.logger.debug('working dir ' + dir_name)
        self.logger.debug('....loading embeddings file')
        self.target_embeddings = self.load_embeddings(target_emb_file)
        self.context_embeddings = self.load_embeddings(context_emb_file)
        self.logger.debug('....reading data')
        playlists_and_tracks = read_data(input_file)
        self.logger.debug('....building corpus')
        self.build_dataset(playlists_and_tracks)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_play_list_constructor(len(self.context_embeddings), self.sampling_table)
        return flatten_list(apply_parallel(process_text, data))

    def build_dataset(self, songs_and_tracks):
        raw_playlists, raw_songs = zip(*songs_and_tracks)
        self.logger.debug('....counting unique playlists')
        count_playlists = collections.Counter(raw_playlists)
        self.logger.debug('....building sampling table')
        self.build_sampling_table(count_playlists)
        self.logger.debug('....building samples')
        self.samples = self.parallel_process_text(songs_and_tracks)
        self.logger.debug('number of samples '+str(len(self.samples)))
        self.logger.debug('....shuffling samples')
        shuffle(self.samples)
        playlists, songs, new_songs, labels = zip(*self.samples)
        self.playlists = np.array(list(playlists))
        self.songs = np.array(list(songs))
        self.new_songs = np.array(list(new_songs))
        self.labels = np.array(list(labels))
        self.logger.debug('....corpus generated')

    def batch_generator(self, n_minibatch):
        batch_size = n_minibatch
        data_playlists = self.playlists
        data_songs = self.songs
        data_new_songs = self.new_songs
        data_labels = self.labels
        while True:
            if data_playlists.shape[0] < batch_size:
                data_playlists = np.hstack([data_playlists, self.playlists])
                data_songs = np.hstack([data_songs, self.songs])
                data_new_songs = np.hstack([data_new_songs, self.new_songs])
                data_labels = np.hstack([data_labels, self.labels])
                if data_playlists.shape[0] < batch_size:
                    continue
            play_lists = data_playlists[:batch_size]
            songs = data_songs[:batch_size]
            new_songs = data_new_songs[:batch_size]
            labels = data_labels[:batch_size]
            data_playlists = data_playlists[batch_size:]
            data_songs = data_songs[batch_size:]
            data_new_songs = data_new_songs[batch_size:]
            data_labels = data_labels[batch_size:]
            yield play_lists, songs, new_songs, labels

    def feed(self, n_minibatch, target_placeholder, context_placeholder, labels_placeholder,
             ones_placeholder, zeros_placeholder, shuffling = False):
        play_lists, songs, labels = self.batch.__next__()
        if shuffling:
            labels = np.random.permutation(labels)
        return {target_placeholder: play_lists,
                context_placeholder: songs,
                labels_placeholder: labels,
                ones_placeholder: np.ones((n_minibatch), dtype=np.int32),
                zeros_placeholder: np.zeros((n_minibatch), dtype=np.int32)
                }

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state

    def load_embeddings(self, emb_file):
        self.logger.debug('....loading playlists embeddings matrix')
        w2v_model = Word2Vec.load(emb_file)
        embeddings = []
        for elem in range(len(w2v_model.wv.vectors)):
            embeddings.append(w2v_model.wv.word_vec(str(elem)))
        self.logger.debug('....embeddings matrix loaded')
        return np.array(embeddings)

    def build_sampling_table(self, count_playlists):
        sampling_factor = 1e-5
        sampling_table = dict()
        total_occurrences = sum(count_playlists.values())
        for playlist in count_playlists:
            playlist_frequency = (1. * count_playlists[playlist]) / total_occurrences
            sampling_table[playlist] = max(0., ((playlist_frequency - sampling_factor) / playlist_frequency) - sqrt(
                sampling_factor / playlist_frequency))
        self.sampling_table = sampling_table

