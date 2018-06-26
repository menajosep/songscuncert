from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')

from classifier.data import *
from classifier.models import *
from classifier.args import *


logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = classifier_data(args.in_file, args.target_emb_file, args.context_emb_file, dir_name, logger)
pickle.dump(d, open(dir_name + "/classifier_data.dat", "wb+"))

# MODEL
#d = pickle.load(open("../class_fits/classifier_data.dat", "rb+"))
#d.batch = d.batch_generator(args.mb)
#m = classifier_model(d, args.mb, sess, dir_name)