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

args, dir_name = parse_args()
os.makedirs(dir_name)

# DATA
d = classifier_data(args.samples, args.in_file, args.target_emb_file, args.context_emb_file, dir_name, logger)
pickle.dump(d, open(dir_name + "/classifier_data.dat", "wb+"))

# MODEL
#d = pickle.load(open("class_fits/classifier_data.dat", "rb+"))
d.batch = d.batch_generator(args.mb)
m = classifier_model(d, dir_name)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    display_step = 1000
    num_steps = 40000

    for step in range(1, num_steps+1):
        # Run optimization op (backprop)
        sess.run(m.train_op, feed_dict=d.feed(m.samples_placeholder, m.labels_placeholder))
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([m.loss_op, m.accuracy], feed_dict=d.feed(m.samples_placeholder, m.labels_placeholder))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(m.accuracy, feed_dict=d.feed(m.samples_placeholder, m.labels_placeholder)))
    # Save the variables to disk.
    save_path = saver.save(sess, dir_name + "model.ckpt")
    print("Model saved in path: %s" % save_path)
