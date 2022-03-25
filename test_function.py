import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('/home/kai/Documents/DATASET/minghong/ps_data/image_0303/IMG_1621/1647214990.627055.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('image.jpg',image)