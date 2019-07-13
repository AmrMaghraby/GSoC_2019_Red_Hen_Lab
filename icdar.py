import glob
import csv
import cv2
import time
import os
import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import compress
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer
import config

tf.app.flags.DEFINE_string('training_data_path','Insert path here','training data')
tf.app.flags.DEFINE_integer('max_image_size',1200,'max size')
tf.app.flags.DEFINE_integer('max_text_size',800,'if it is greater that this it will be resiezed')
tf.app.flags.DEFINE_float('min_crop_side_ratio',0.1,'')
tf.app.flags.DEFINE_string('geometry','RBOX','')

