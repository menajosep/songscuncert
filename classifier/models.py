from utils import *
import tensorflow as tf


class classifier_model():
    def __init__(self, sess, logdir):
        self.logdir = logdir
        self.sess = sess

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.samples_placeholder = tf.placeholder(tf.float32, shape=[None, 300])
                self.playlists_placeholder = tf.placeholder(tf.float32, shape=[None, 100])
                self.seeds_placeholder = tf.placeholder(tf.float32, shape=[None, 3000])
                self.labels_placeholder = tf.placeholder(tf.float32)

            num_classes = 1

            # layer_seeds = tf.layers.dense(inputs=self.seeds_placeholder, units=2000, activation=tf.nn.relu)
            # layer_seeds_2 = tf.layers.dense(inputs=layer_seeds, units=1500, activation=tf.nn.relu)
            # layer_seeds_3 = tf.layers.dense(inputs=layer_seeds_2, units=1000, activation=tf.nn.relu)
            # layer_seeds_4 = tf.layers.dense(inputs=layer_seeds_3, units=300, activation=tf.nn.relu)
            # layer_seeds_5 = tf.layers.dense(inputs=layer_seeds_4, units=100, activation=tf.nn.relu)
            # layer_songs = tf.layers.dense(inputs=self.samples_placeholder, units=300, activation=tf.nn.relu)
            # layer_songs_2 = tf.layers.dense(inputs=layer_songs, units=300, activation=tf.nn.relu)
            # layer_songs_3 = tf.layers.dense(inputs=layer_songs_2, units=100, activation=tf.nn.relu)
            # layer_all = tf.concat([self.playlists_placeholder, layer_songs_3, layer_seeds_5], axis=1)
            # layer4 = tf.layers.dense(inputs=layer_all, units=64, activation=tf.nn.relu)
            # out_layer = tf.layers.dense(inputs=layer4, units=num_classes)

            layer_input = tf.concat([self.playlists_placeholder,
                                     self.samples_placeholder,
                                     self.seeds_placeholder], axis=1)
            layer_2 = tf.layers.dense(inputs=layer_input, units=8000, activation=tf.nn.relu)
            layer_3 = tf.layers.dense(inputs=layer_2, units=8000, activation=tf.nn.relu)
            layer_4 = tf.layers.dense(inputs=layer_3, units=4000, activation=tf.nn.relu)
            layer_5 = tf.layers.dense(inputs=layer_4, units=2000, activation=tf.nn.relu)
            layer_6 = tf.layers.dense(inputs=layer_5, units=512, activation=tf.nn.relu)
            layer_7 = tf.layers.dense(inputs=layer_6, units=64, activation=tf.nn.relu)

            out_layer = tf.layers.dense(inputs=layer_7, units=num_classes, activation=tf.nn.relu)



            # Construct model
            logits = out_layer
            predicted = tf.nn.sigmoid(logits)

            # Define loss and optimizer
            learning_rate = 1e-8
            self.loss_op = tf.reduce_mean(tf.losses.mean_squared_error(self.labels_placeholder, predicted))
            #self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #    logits=logits, labels=self.labels_placeholder))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

            # Evaluate model (with test logits, for dropout to be disabled)
            self.correct_pred = tf.equal(tf.round(predicted), self.labels_placeholder)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            with self.sess.as_default():
                tf.global_variables_initializer().run()
            # Create a summary to monitor cost tensor
            tf.summary.scalar("loss", self.loss_op)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)
            summaries = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
            self.saver = tf.train.Saver()
