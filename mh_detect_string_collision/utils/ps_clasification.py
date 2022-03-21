from importlib import import_module
import cv2
from numpy import imag
from display import visualization
import numpy as np
import time
from sklearn.cluster import DBSCAN
from utils import ps_status_check
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
# from imantics import Polygons, Mask
def switch_xy(points):
    points = np.flip(points, 1)
    return points

def dbscan(image, threadhold_value, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_image = gray[config.crop_y[0]:config.crop_y[1], config.crop_x[0]:config.crop_x[1]]
    crop_image_color = image[config.crop_y[0]:config.crop_y[1], config.crop_x[0]:config.crop_x[1]]
    # optimize_line
    # crop_image = cv2.resize(crop_image, (int(crop_image.shape[1]/2), int(crop_image.shape[0]/2)), interpolation = cv2.INTER_AREA)
    _, converted_image = cv2.threshold(crop_image, threadhold_value, 255, cv2.THRESH_BINARY)
    pixel_list = np.argwhere(converted_image == 255)
    clustering = DBSCAN(eps=6, min_samples=30).fit(pixel_list)
    label = clustering.labels_

    collision_occurred = False
    for label_index in range(label.min(),label.max(),1):
        current_ps_pixel_data = np.take(pixel_list, 
                        (np.where(label == label_index)[0]), 
                        0)
        current_ps_pixel_data = switch_xy(current_ps_pixel_data)
        new_array = np.zeros((current_ps_pixel_data[:,1].max()+10,current_ps_pixel_data[:,0].max()+10))
        start_time = time.time()
        # new_array[current_ps_pixel_data[:,1],current_ps_pixel_data[:,0]] = 1
        # img = Image.new('L', (gray.shape[1], gray.shape[0]), 0)
        # ImageDraw.Draw(img).polygon(current_ps_pixel_data, outline=1, fill=1)
        # mask = np.array(img)
        # polygons = current_ps_pixel_data[np.array(list(alpha_shape(current_ps_pixel_data,1000)))]
        polygons = current_ps_pixel_data[ConvexHull(current_ps_pixel_data).vertices]
        # polygons = Mask(new_array).polygons()
        # polygons = polygons.polygons[0].reshape(-1,2)
        end_time = time.time()
        print(end_time - start_time)
        # continue
        for index, point in enumerate(polygons):
            if index < len(polygons) - 2:
                crop_img = cv2.line(crop_image_color, tuple(point), tuple(polygons[index+1]), [0,0,255], 1)
                # point = point[0]
                # crop_img = cv2.line(crop_image_color, tuple([point[1],point[0]]), tuple([polygons[index+1][0][1], polygons[index+1][0][0]]), [0,0,255], 1)
            else:
                crop_img = cv2.line(crop_image_color, tuple(point), tuple(polygons[0]), [0,0,255], 1)
        cv2.imshow('image', crop_img)
        cv2.waitKey(1)
        continue
        heigh_px_list = current_ps_pixel_data[:,0]
        ps_heigh =  heigh_px_list.max() -  heigh_px_list.min()
        if ps_heigh*1.5 > crop_image.shape[0]:
            current_ps_check = ps_status_check.PS_Check(current_ps_pixel_data,config)
            collision, draw_points = current_ps_check.check_two_side()
            draw_index = int(len(draw_points)/2)
            if collision:
                collision_occurred = True
                print(label_index)
                start_point = np.array(draw_points[draw_index])+ np.array([config.crop_y[0],config.crop_x[0]])
                # end_point = tuple(np.array(draw_points[-1])+ np.array([crop_value[1][0],crop_value[0][0]]))
                image = cv2.circle(image, (start_point[1],start_point[0]), 30, [0,255,255], 3)
    image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation = cv2.INTER_AREA)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    return collision_occurred