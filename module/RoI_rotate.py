import sys
sys.path.append("..")
import numpy as np
import cv2
import tensorflow as tf
import math
import config
import os

from stn import spatial_transformer_network as transformer

class RoIRotate(object):
        
