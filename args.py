import argparse
import os
import time
def parse_args():
        parser = argparse.ArgumentParser(description="run exponential family embeddings on songs and playlists")

        parser.add_argument('--K', type=int, default=300,
                            help='Number of dimensions. Default is 300.')

        parser.add_argument('--cs', type=int, default=50,
                            help='Size of the context. Default is 50.')

        parser.add_argument('--n_epochs', type=int, default=10,
                            help='Number of epochs. Default is 10.')

        parser.add_argument('--ns', type=int, default=20,
                            help='Number of negative samples. Default is 20.')

        parser.add_argument('--mb', type=int, default=5000,
                            help='Minibatch size. Default is 5000.')

        parser.add_argument('--in_file', type=str, default=None,
                            help='input file')

        parser.add_argument('--target_emb_file', type=str, default=None,
                            help='pretrained embedings file for target items')

        parser.add_argument('--context_emb_file', type=str, default=None,
                            help='pretrained embedings file for context items')

        args =  parser.parse_args()
        dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

        return args, dir_name
