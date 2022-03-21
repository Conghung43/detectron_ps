import numpy as np

class Config():
    def __init__(self, image_shape):
        self.get_cut_pattent = (np.array([0.6])*image_shape[0]).astype(int)
        self.crop_y = (np.array([0.33, 0.5])*image_shape[0]).astype(int)
        self.crop_x = (np.array([0.00,1])*image_shape[1]).astype(int)
        self.threadhold_from_peak  = 70
        self.peak_distance = 100
        self.default_value = 1000
        self.valid_angle = 10
        self.head_tail_diff_ratio = 1.5

    def switch_color(color_code):
        print('switch color')
