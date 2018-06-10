import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import pickle
import numpy as np
import csv

variational_data = pickle.load(open('fits/playlists_songs_300D/variational.dat', 'rb'))
playlist_embeddings = variational_data['rhos']
playlist_sigmas = variational_data['sigma_rhos']
discard_noise_indexes = np.where(playlist_sigmas > 0.05)
good_embeddings = np.delete(playlist_embeddings, discard_noise_indexes, axis=0)

# type = 'raw'|filtered
type = 'raw'
if type == 'raw':
    embeddings = playlist_embeddings
else:
    embeddings = good_embeddings
out_dir = type


tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embeddings.shape)
set_x = tf.assign(X, place, validate_shape=False)

sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embeddings})

os.makedirs(out_dir + '/' + type+'_emb_viz.log')
# write labels
with open(out_dir + '/' + type+'_emb_viz.log' + '/'+type+'_metadata.tsv', 'w') as f:
    line_out = "title\n"
    f.write(line_out)
    with open('fits/playlists_songs_300D/dict_id_ntitle.csv', 'r') as dict_file:
        dictionary_reader = csv.reader(dict_file, delimiter=',', quotechar='|')
        for row in dictionary_reader:
            if type == 'raw':
                if row[0] != 'id':
                    line_out = "%s\n" % row[1]
                    f.write(line_out)
            else:
                if row[0] != 'id' and not np.isin(int(row[0]), discard_noise_indexes):
                    line_out = "%s\n" % row[1]
                    f.write(line_out)

# create a TensorFlow summary writer
#summary_writer = tf.summary.FileWriter(out_dir + '/' + 'filtered_emb_viz.log', sess.graph)
summary_writer = tf.summary.FileWriter(out_dir + '/' + type+'_emb_viz.log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(out_dir + '/' + type+'_emb_viz.log', '../../'+type+'_metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
saver.save(sess, os.path.join(out_dir + '/' + type+'_emb_viz.log', type+"_model.ckpt"))