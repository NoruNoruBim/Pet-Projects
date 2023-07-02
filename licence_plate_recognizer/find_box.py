import os
import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.visualize import display_images
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.model import log

import custom 


'''
github
here is removed code
this code uses pre-trained mask r-cnn to find licence plate on the image
'''
