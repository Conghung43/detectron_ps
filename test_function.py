import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# ax = plt.axes(projection ="3d")

# image = cv2.imread('dummy_data/objects_collection.jpg')
# f =  open('dummy_data/objects_collection.npy', 'rb')
# objects_collection = np.load(f, allow_pickle=True)
# objects_collection = np.array([objects_collection[index] for index in range(0, len(objects_collection),50)])
# x,y,z = [objects_collection[:,0], 
#                     objects_collection[:,1], 
#                     objects_collection[:,2]]
# print(len(objects_collection))
# sctt = ax.scatter3D(x,y,z,
#                     alpha = 0.8,
#                     c = (x + y + z),
#                     )
# ax.set_xlim([0,255])
# ax.set_ylim([0,255])
# ax.set_zlim([0,255])
# plt.show()

def correct_polygon(raw_polygon):
    print('readyt')

# f =  open('dummy_data/auto_detect_polygon.npy', 'rb')
# raw_polygon = np.load(f, allow_pickle=True)
# correct_polygon(raw_polygon)

#finding hsv range of target object(pen)
import cv2
import time
import glob
import os
import scipy.ndimage as ndimage
from numpy import arange
# from detectron2.utils.visualizer import ColorMode, Visualizer
# A required callback method that goes into the trackbar function.
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


def curve_fit(data):
    x,y = data[:,0], data[:,1]
    popt, _  = curve_fit(objective_2, x, y)
    a, b, c = popt
    x_line = arange(min(x), max(x), (max(x) - min(x))/len(data))
    y_line = objective_2(x_line, a, b, c)
    
    pyplot.plot(x_new, y_line, '--', color='red')
# masks = create_masks(frame)

# frame_visualizer = Visualizer(frame, self.metadata)
# mask_layers = frame_visualizer._convert_masks(masks)

# # Initializing the webcam feed.
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# Create a window named trackbars.
# cv2.namedWindow("Trackbars")

# # Now create 6 trackbars that will control the lower and upper range of 
# # H,S and V channels. The Arguments are like this: Name of trackbar, 
# # window name, range,callback function. For Hue the range is 0-179 and
# # for S,V its 0-255.
# cv2.createTrackbar("L - H", "Trackbars", 53, 179, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 62, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 97, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
# frame = cv2.imread('dummy_data/objects_collection.jpg')
# while True:
    
#     # Start reading the webcam feed frame by frame.
#     # ret, frame = cap.read()
#     # if not ret:
#     #     break
#     # Flip the frame horizontally (Not required)
#     frame = cv2.flip( frame, 1 ) 
    
#     # Convert the BGR image to HSV image.
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Get the new values of the trackbar in real time as the user changes 
#     # them
#     l_h = cv2.getTrackbarPos("L - H", "Trackbars")
#     l_s = cv2.getTrackbarPos("L - S", "Trackbars")
#     l_v = cv2.getTrackbarPos("L - V", "Trackbars")
#     u_h = cv2.getTrackbarPos("U - H", "Trackbars")
#     u_s = cv2.getTrackbarPos("U - S", "Trackbars")
#     u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
#     # Set the lower and upper HSV range according to the value selected
#     # by the trackbar
#     lower_range = np.array([l_h, l_s, l_v])
#     upper_range = np.array([u_h, u_s, u_v])
    
#     # Filter the image and get the binary mask, where white represents 
#     # your target color
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#     # create_separate_mask(mask)
#     # You can also visualize the real part of the target color (Optional)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
    
#     # Converting the binary mask to 3 channel image, this is just so 
#     # we can stack it with the others
#     mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
#     # stack the mask, orginal frame and the filtered result
#     stacked = np.hstack((mask_3,frame,res))
    
#     # Show this stacked frame at 40% of the size.
#     cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
#     # If the user presses ESC then exit the program
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
    
#     # If the user presses `s` then print this array.
#     if key == ord('s'):
        
#         thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
#         print(thearray)
        
#         # Also save this array as penval.npy
#         np.save('hsv_value',thearray)
#         break
    
# Release the camera & destroy the windows.    
# cap.release()
# cv2.destroyAllWindows()