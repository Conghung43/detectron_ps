import time
import cv2
import os
import json
import numpy as np

def save_image_gen_json(image, polygon_layers,path):
    current_time = str(time.time())+'IMG_1540'
    save_path = f'{path}{current_time}.jpg'
    cv2.imwrite(save_path, image)
    bytes_size = os.path.getsize(save_path) 
    filename = f'{current_time}.jpg'
    title_key = f'{filename}{bytes_size}'
    region_atributes = {"object":{"ps":True}}

    json_data = {}
    objects_data = []
    for mask_polygon in polygon_layers:
        mask_polygon = np.array(mask_polygon)
        # mask_polygon = np.reshape(mask_polygon,(-1,2)).astype(int)
        data = {}
        all_points_x = mask_polygon[:,0]
        if all_points_x.max() - all_points_x.min() < image.shape[1]*0.5:
            continue
        all_points_x = all_points_x.tolist()
        all_points_y = mask_polygon[:,1].tolist()
        shape_attributes = {}
        shape_attributes['name'] = 'polygon'
        shape_attributes['all_points_x'] = all_points_x
        shape_attributes['all_points_y'] = all_points_y
        data['shape_attributes']=shape_attributes
        data['region_atributes']=region_atributes
        objects_data.append(data)
    json_data['filename'] = filename
    json_data['size'] = bytes_size
    json_data['regions'] = objects_data
    json_data['file_atributes'] = {}
    return title_key,json_data




def write_json(json_data, path):
    output_json_file_path = path + "new.json"
    output_file = open(output_json_file_path, 'w')
    json.dump(json_data, output_file)