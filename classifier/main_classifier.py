from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')

from classifier.data import *
from models import *
from classifier.args import *


logger = get_logger()

args, dir_name = parse_args()
os.makedirs(dir_name)
sess = ed.get_session()

# DATA
d = classifier_data(args.in_file, args.target_emb_file, args.context_emb_file, dir_name, logger)
pickle.dump(d, open(dir_name + "/classifier_data.dat", "wb+"))