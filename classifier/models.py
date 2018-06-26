from utils import *


class classifier_model():
    def __init__(self, d, logdir):
        self.logdir = logdir

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.samples_placeholder = tf.placeholder(tf.int32)
                self.labels_placeholder = tf.placeholder(tf.float32)

            # Network Parameters
            n_hidden_1 = 256  # 1st layer number of neurons
            n_hidden_2 = 256  # 2nd layer number of neurons
            num_input = 300  # MNIST data input (img shape: 28*28)
            num_classes = 1  # MNIST total classes (0-9 digits)

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([num_classes]))
            }

            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.add(tf.matmul(self.samples_placeholder, weights['h1']), biases['b1'])
            # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # Output fully connected layer with a neuron for each class
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

            # Construct model
            logits = out_layer

            # Define loss and optimizer
            learning_rate = 0.1
            loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self.labels_placeholder))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(loss_op)

            # Evaluate model (with test logits, for dropout to be disabled)
            self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
