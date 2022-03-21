import cv2
from initiation import init_config
import time
import numpy as np
import utils.find_white_ps as find_white_ps
from display import main_display
from utils import ps_clasification
from config import read_config

system_config = init_config.get_config_data()
image = cv2.imread('testing_set/IMG_1068_modified.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
config = read_config.Config(gray.shape)
s_time= time.time()
string_property = []
for pattent_xaxis in config.get_cut_pattent:
    main_display.draw_cut_line(image, pattent_xaxis,[0,255,0])
    data_list = gray[pattent_xaxis][config.crop_x[0]:config.crop_x[1]]
    average_threadhold, peak_pixel = find_white_ps.get_string_property(data_list,
                                            config.threadhold_from_peak, 
                                            config.peak_distance,
                                            config.crop_x[0])
    # main_display.show_gray_image_threadhold(gray, average_threadhold)
    ps_clasification.dbscan(image, average_threadhold, config)
main_display.draw_detected_white_line(image, string_property)
e_time = time.time()
print(e_time - s_time)
