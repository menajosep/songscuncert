from utils import *
from random import shuffle
import collections



class bayessian_bern_emb_data():
    def __init__(self, input_file, ns, n_minibatch, L, K, cs, dir_name, logger):
        self.logger = logger
        self.logger.debug('initializing bayessian_bern_emb_data with file '+input_file)
        self.logger.debug('neg sampling '+str(ns))
        self.logger.debug('n_minibatch '+str(n_minibatch))
        self.logger.debug('songs dictionary size of '+str(L))
        self.logger.debug('dimesion of embeddings '+str(K))
        self.logger.debug('working dir '+dir_name)
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        self.K = K
        self.cs = cs
        self.dir_name = dir_name
        self.logger.debug('....reading data')
        songs_and_tracks = read_data(input_file)
        self.logger.debug('....building corpus')
        self.build_dataset(songs_and_tracks)
        #self.batch = self.batch_generator()
        self.N = len(self.playlists)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_play_list_constructor(self.ns, self.dictionary, self.cs)
        return flatten_list(apply_parallel(process_text, data))

    def build_dataset(self, songs_and_tracks):
        raw_playlists, raw_songs = zip(*songs_and_tracks)
        self.logger.debug('....counting unique playlists')
        count_playlists = collections.Counter(raw_playlists)
        self.L_target = len(count_playlists.keys())
        self.logger.debug('number of unique playlists '+str(self.L_target))
        self.logger.debug('....counting unique songs')
        raw_songs = flatten_list(raw_songs)
        count_songs = [['UNK', -1]]
        count_songs.extend(collections.Counter(raw_songs).most_common(self.L - 1))
        self.logger.debug('....building songs dictionary')
        self.dictionary = dict()
        self.counter = dict()
        for song, _ in count_songs:
            self.dictionary[song] = len(self.dictionary)
        self.L_context = self.L
        self.logger.debug('size of songs dictionary '+str(self.L_context))
        self.logger.debug('....building samples')
        self.samples = self.parallel_process_text(songs_and_tracks)
        self.logger.debug('number of samples '+str(len(self.samples)))
        self.logger.debug('....shuffling samples')
        shuffle(self.samples)
        playlists, songs, labels = zip(*self.samples)
        self.playlists = np.array(list(playlists))
        self.songs = np.array(list(songs))
        self.labels = np.array(list(labels))
        self.logger.debug('....corpus generated')

    def batch_generator(self):
        batch_size = self.n_minibatch
        data_target = self.playlists
        data_context = self.songs
        data_labels = self.labels
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, self.playlists])
                data_context = np.hstack([data_context, self.songs])
                data_labels = np.hstack([data_labels, self.labels])
                if data_target.shape[0] < batch_size:
                    continue
            play_lists = data_target[:batch_size]
            songs = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield play_lists, songs, labels

    def feed(self, target_placeholder, context_placeholder, labels_placeholder,
             ones_placeholder, zeros_placeholder, shuffling = False):
        play_lists, songs, labels = self.batch.__next__()
        if shuffling:
            labels = np.random.permutation(labels)
        return {target_placeholder: play_lists,
                context_placeholder: songs,
                labels_placeholder: labels,
                ones_placeholder: np.ones((self.n_minibatch), dtype=np.int32),
                zeros_placeholder: np.zeros((self.n_minibatch), dtype=np.int32)
                }

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state