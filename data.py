from utils import *
from random import shuffle
import collections
import logging



class bayessian_bern_emb_data():
    def __init__(self, input_file, ns, n_minibatch, L, K, cs, dir_name):
        logging.debug(f'initializing bayessian_bern_emb_data with file {input_file}')
        logging.debug(f'neg sampling {ns} ')
        logging.debug(f'n_minibatch {n_minibatch} ')
        logging.debug(f'songs dictionary size of {L}')
        logging.debug(f'dimesion of embeddings {K}')
        logging.debug(f'working dir {dir_name}')
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        self.K = K
        self.cs = cs
        self.dir_name = dir_name
        logging.debug(f'....reading data')
        songs_and_tracks = read_data(input_file)
        logging.debug(f'....building corpus')
        self.build_dataset(songs_and_tracks)
        #self.batch = self.batch_generator()
        self.N = len(self.playlists)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_play_list_constructor(self.ns, self.dictionary, self.cs)
        return flattenlist(apply_parallel(process_text, data))

    def build_dataset(self, songs_and_tracks):
        raw_playlists, raw_songs = zip(*songs_and_tracks)
        logging.debug(f'....counting unique playlists')
        count_playlists = collections.Counter(raw_playlists)
        self.L_target = len(count_playlists.keys())
        logging.debug(f'number of unique playlists {self.L_target}')
        logging.debug(f'....counting unique songs')
        raw_songs = flattenlist(raw_songs)
        count_songs = [['UNK', -1]]
        count_songs.extend(collections.Counter(raw_songs).most_common(self.L - 1))
        logging.debug(f'....building songs dictionary')
        self.dictionary = dict()
        self.counter = dict()
        for song, _ in count_songs:
            self.dictionary[song] = len(self.dictionary)
        self.L_context = self.L
        logging.debug(f'size of songs dictionary {self.L_context}')
        logging.debug(f'....building samples')
        self.samples = self.parallel_process_text(songs_and_tracks)
        logging.debug(f'number of samples {len(self.samples)}')
        logging.debug(f'....shuffling samples')
        shuffle(self.samples)
        playlists, songs, labels = zip(*self.samples)
        self.playlists = np.array(list(playlists))
        self.songs = np.array(list(songs))
        self.labels = np.array(list(labels))
        logging.debug(f'....corpus generated')

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