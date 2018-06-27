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
            n_hidden_1 = 256  # 1st layer number of neurons
            n_hidden_2 = 256  # 2nd layer number of neurons
            n_hidden_3 = 64  # 3rd layer number of neurons
            num_input = 300  # MNIST data input (img shape: 28*28)
            num_classes = 1  # MNIST total classes (0-9 digits)

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name="h1"),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="h2"),
                'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name="h3"),
                'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]), name="hout")
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
                'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="b2"),
                'b3': tf.Variable(tf.random_normal([n_hidden_3]), name="b3"),
                'out': tf.Variable(tf.random_normal([num_classes]), name="bout")
            }

            # Hidden fully connected layer with 256 neurons
            layer_1 = tf.add(tf.matmul(self.samples_placeholder, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Hidden fully connected layer with 64 neurons
            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.relu(layer_3)
            # Output fully connected layer with a neuron for each class
            out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

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
