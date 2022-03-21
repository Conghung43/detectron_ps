import cv2
from numpy import imag
from display import visualization
import numpy as np
def draw_cut_line(image, line_y_postion, color):
    heigh, width = image.shape[:2]
    cv2.line(image, (0,int(line_y_postion)), (width,int(line_y_postion)), color, 1)
    visualization.write_image('display', 'test_cut_line', image)

def draw_detected_white_line(image, string_property_list):
    for string_property in string_property_list:
        for index in range(len(string_property.string_position_list)):
            if index > 0:
                pos, width = string_property.get(index)
                pre_pos, _ = string_property.get(index-1)
                image = cv2.line(image, pos, pre_pos, [0,255,0], width)
    visualization.write_image('display', 'test_cut_line', image)

# def show_gray_image_threadhold(gray, threadhold_value):
#     visualization.write_image('display', 'test_cut_line', converted_image)