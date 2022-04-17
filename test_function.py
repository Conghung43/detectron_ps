#finding hsv range of target object(pen)
import numpy as np
import cv2
import time
import glob
import os
import scipy.ndimage as ndimage
from numpy import arange

def nothing(x):
    pass
l_h, l_s, l_v,u_h, u_s, u_v = [53,63,97,179,255,255]

def create_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    label_im, nb_labels = ndimage.label(mask) 
    masks = []

    for i in range(nb_labels):

        # create an array which size is same as the mask but filled with 
        # values that we get from the label_im. 
        # If there are three masks, then the pixels are labeled 
        # as 1, 2 and 3.

        mask_compare = np.full(np.shape(label_im), i+1) 

        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int) 
        masks.append(separate_mask)
        continue
        # replace 1 with 255 for visualization as rgb image

        separate_mask[separate_mask == 1] = 255 
    return masks

def objective_2(x, b, c, a):
	return (b * x) + (c * x**2) + a 