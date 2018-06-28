from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import matplotlib
matplotlib.use('Agg')

from classifier.data import *
from classifier.models import *
from classifier.args import *


logger = get_logger()

args, dir_name = parse_args_model()
os.makedirs(dir_name)

# MODEL
logger.debug('load data set from disk')
d = pickle.load(open(args.in_file, "rb+"))
samples = np.array(d['samples'])
labels = np.array(d['labels'])
batch = batch_generator(args.mb, samples, labels)

logger.debug('init training')
# Start training
with tf.Session() as sess:
    m = classifier_model(sess, dir_name)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Run the initializer
    sess.run(init)
    display_step = 1000
    num_steps = 400000

    for step in range(1, num_steps+1):
        # Run optimization op (backprop)
        feed_dict = feed(batch, m.samples_placeholder, m.labels_placeholder)
        sess.run(m.train_op, feed_dict=feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([m.loss_op, m.accuracy], feed_dict=feed_dict)
            logger.debug("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    logger.debug("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    logger.debug("Testing Accuracy:", \
        sess.run(m.accuracy, feed_dict=feed(batch, m.samples_placeholder, m.labels_placeholder)))
    # Save the variables to disk.
    save_path = m.saver.save(sess, dir_name + "/model.ckpt")
    logger.debug("Model saved in path: %s" % save_path)
