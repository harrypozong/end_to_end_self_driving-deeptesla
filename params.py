import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf

#### CONST
## Initialize Constant
flags = tf.app.flags
FLAGS = flags.FLAGS

## Nvida's camera format
flags.DEFINE_integer('img_h', 64, 'The image height.')
flags.DEFINE_integer('img_w', 64, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')
flags.DEFINE_integer('batch_size', 32, 'The number of channels.')
flags.DEFINE_integer('train_batch_per_epoch', 1, 'The number of channels.')

## Fix random seed for reproducibility
np.random.seed(42)

## Path
data_dir = 'C:/Users/ccc/Downloads/udacity-capstone-deeptesla-master/epochs'
out_dir = 'C:/Users/ccc/Downloads/udacity-capstone-deeptesla-master//output'
model_dir ='C:/Users/ccc/Downloads/udacity-capstone-deeptesla-master//models'
