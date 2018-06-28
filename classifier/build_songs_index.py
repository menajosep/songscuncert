import argparse
import os
import pickle
from panns import *
import numpy.linalg as la
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="build a panns index for valid embeddings")

    parser.add_argument('--songs_emb_file', type=str, default=None,
                        help='pretrained embedings file for songs')

    parser.add_argument('--sigmas_file', type=str, default=None,
                        help='sigma values for the songs')

    parser.add_argument('--sigma_threshold', type=float, default=1.,
                        help='sigma value setting the max value for sigmas for good embeddings')

    args = parser.parse_args()
    dir_name = 'songs_index'

    return args, dir_name

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


logger = get_logger()
args, dir_name = parse_args()
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
# create an index of Euclidean distance
p = PannsIndex(dimension=300, metric='angular')
logger.info("load sigmas")
sigmas = pickle.load(open(args.sigmas_file, 'rb'))
logger.info("load embeddings")
song_embeddings = load_embeddings(args.songs_emb_file)
valid_songs_vocab = {}
counter = 0
for song in tqdm(song_embeddings.wv.vocab):
    sigma = sigmas[song]
    if sigma < args.sigma_threshold:
        embedding = song_embeddings.wv.word_vec(song)
        embedding = embedding / la.norm(embedding)
        #logger.debug("Vector norm:" + str(la.norm(embedding)))
        p.add_vector(embedding)
        valid_songs_vocab[counter] = song
        counter += 1
# enable the parallel building mode
p.parallelize(True)
logger.info("build the index")
# build an index of 128 trees and save to a file
p.build(128)

random = gaussian_vector(300)
results = p.query(random,10)
p.save(dir_name+'/songs_'+str(args.sigma_threshold)+'.idx')
pickle.dump(valid_songs_vocab, open(dir_name+'/valid_songs_vocab.p', "wb+"))
logger.info("Done")



