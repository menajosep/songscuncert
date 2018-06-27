from random import shuffle

from classifier.utils import *


class classifier_data():
    def __init__(self, samples, input_file, target_emb_file, context_emb_file, dir_name, logger):
        self.logger = logger
        self.logger.debug('initializing classifier_data with file ' + input_file)
        self.logger.debug('working dir ' + dir_name)
        self.logger.debug('....loading embeddings file')
        self.target_emb_file = target_emb_file
        self.context_emb_file = context_emb_file
        self.logger.debug('....reading data')
        playlists_and_tracks = read_data(input_file)
        self.logger.debug('....building corpus')
        self.samples = samples
        self.build_dataset(playlists_and_tracks)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_play_list_constructor(self.target_emb_file, self.context_emb_file)
        return flatten_list(apply_parallel(process_text, data))

    def build_dataset(self, playlists_and_tracks):
        self.logger.debug('....building sampling table')
        self.build_sampling_table(playlists_and_tracks)
        self.logger.debug('....building samples')
        self.samples = self.parallel_process_text(self.sampling_playlists)
        self.logger.debug('number of samples '+str(len(self.samples)))
        self.logger.debug('....shuffling samples')
        shuffle(self.samples)
        samples, labels = zip(*self.samples)
        self.samples = np.array(list(samples))
        self.labels = np.array(list(labels))
        self.logger.debug('....corpus generated')

    def batch_generator(self, n_minibatch):
        batch_size = n_minibatch
        data_samples = self.samples
        data_labels = self.labels
        while True:
            if data_samples.shape[0] < batch_size:
                data_samples = np.concatenate([data_samples, self.samples])
                data_labels = np.concatenate([data_labels, self.labels])
                if data_samples.shape[0] < batch_size:
                    continue
            samples = data_samples[:batch_size]
            labels = data_labels[:batch_size]
            data_samples = data_samples[batch_size:]
            data_labels = data_labels[batch_size:]
            yield samples, labels

    def feed(self, samples_placeholder, labels_placeholder, shuffling = False):
        samples, labels = self.batch.__next__()
        if shuffling:
            labels = np.random.permutation(labels)
        return {samples_placeholder: samples,
                labels_placeholder: labels}

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state



    def build_sampling_table(self, playlists_and_tracks):
        playlists = []
        while len(playlists) < self.samples:
            playlist_id = random.randint(0, len(playlists_and_tracks) - 1)
            songs = playlists_and_tracks[playlist_id][1]
            if len(songs) > 1:
                playlists.append(playlists_and_tracks[playlist_id])
        self.sampling_playlists = playlists

