from distutils.command.config import config
import numpy as np

class Config():
    def __init__(self, image_shape):
        self.get_cut_pattent = (np.array([0.6])*image_shape[0]).astype(int)
        self.crop_y = (np.array([0.33, 0.5])*image_shape[0]).astype(int)
        self.crop_x = (np.array([0.00,1])*image_shape[1]).astype(int)
        self.shortest_distance  = 10
        self.predict_range_base_fov = 0.5
        self.confirm_number_ps = False
        self.number_of_ps = 15

    def switch_color(color_code):
        print('switch color')
