from utils import *


class classifier_model():
    def __init__(self, sess, d, logdir):
        self.logdir = logdir
        self.sess = sess

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.samples_placeholder = tf.placeholder(tf.float32)
                self.labels_placeholder = tf.placeholder(tf.float32)

            # Network Parameters
            n_hidden_1 = 500  # 1st layer number of neurons
            n_hidden_2 = 500  # 2nd layer number of neurons
            n_hidden_3 = 300  # 3rd layer number of neurons
            n_hidden_4 = 64  # 3rd layer number of neurons
            num_classes = 1  # MNIST total classes (0-9 digits)

            layer1 = tf.layers.dense(inputs=self.samples_placeholder, units=n_hidden_1, activation=tf.nn.relu)
            layer2 = tf.layers.dense(inputs=layer1, units=n_hidden_2, activation=tf.nn.relu)
            layer3 = tf.layers.dense(inputs=layer2, units=n_hidden_3, activation=tf.nn.relu)
            layer4 = tf.layers.dense(inputs=layer3, units=n_hidden_4, activation=tf.nn.relu)
            out_layer = tf.layers.dense(inputs=layer4, units=num_classes)

            # Construct model
            logits = out_layer

            # Define loss and optimizer
            learning_rate = 0.01
            self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self.labels_placeholder))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

            # Evaluate model (with test logits, for dropout to be disabled)
            self.correct_pred = tf.equal(tf.round(logits), self.labels_placeholder)
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
