import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.contrib.tensorboard.plugins import projector
import os
import pickle
import numpy as np
import csv

type = 'gensim'

emb_file_name = 'fits/playlists_songs_300D/wv_model_titles_MPD_final'
dict_file_name = 'fits/playlists_songs_300D/dict_id_ntitle_final.csv'

w2v_model = Word2Vec.load(emb_file_name)
embeddings = []

# write labels and get embeddings
i = 0
out_dir = type
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
# write labels
with open(out_dir + '/'+type+'_metadata.tsv', 'w') as f:
    with open(dict_file_name, 'r') as dict_file:
        dictionary_reader = csv.reader(dict_file, delimiter=',')
        for row in dictionary_reader:
            if row[0] != 'id':
                if str(i) != row[0]:
                    print(i)
                i += 1
                line_out = "%s\n" % row[1]
                f.write(line_out)
                embeddings.append(w2v_model.wv.word_vec(row[0]))
embeddings = np.array(embeddings)

# build the tf graph
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embeddings.shape)
set_x = tf.assign(X, place, validate_shape=False)

# run it with the learned embeddings
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embeddings})


# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(out_dir, '../'+type+'_metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
saver.save(sess, os.path.join(out_dir, type+"_model.ckpt"))