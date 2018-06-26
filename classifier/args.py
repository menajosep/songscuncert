import argparse
import time
def parse_args():
        parser = argparse.ArgumentParser(description="run classifier on songs and playlists")

        parser.add_argument('--in_file', type=str, default=None,
                            help='input file')

        parser.add_argument('--target_emb_file', type=str, default=None,
                            help='pretrained embedings file for target items')

        parser.add_argument('--context_emb_file', type=str, default=None,
                            help='pretrained embedings file for context items')

        parser.add_argument('--n_epochs', type=int, default=10,
                            help='Number of epochs. Default is 10.')

        parser.add_argument('--mb', type=int, default=5000,
                            help='Minibatch size. Default is 5000.')

        args = parser.parse_args()
        dir_name = 'class_fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

        return args, dir_name
