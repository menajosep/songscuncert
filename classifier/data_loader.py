from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from classifier.data import *
from classifier.args import *


logger = get_logger()

args, dir_name = parse_args_data()
os.makedirs(dir_name)

# DATA
d = classifier_data(args.samples, args.in_file, args.target_emb_file, args.context_emb_file, dir_name, logger)
