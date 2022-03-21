import numpy as np

# f =  open('testing_set/{}.npy'.format('first_array'), 'rb')
# string_pixel_list = np.load(f, allow_pickle=True)
# diff_ratio = 1.5
# string_default_width = 10
# def main(string_pixel_list, diff_ratio):

class PS_Check():
    def __init__(self, ps_pixels, config):
        self.ps_pixels = ps_pixels
        self.config = config
        # self.check_two_side()

    def check_two_side(self):
        heigh_px_list = self.ps_pixels[:,0]
        width_px_list = self.ps_pixels[:,1]
        ps_heigh =  heigh_px_list.max() -  heigh_px_list.min()
        ps_width =  width_px_list.max() -  width_px_list.min()
        collision = False
        draw_point = []
        if ps_heigh > ps_width:
            self.ratio_list = []
            for index in range(2):
                self.three_position = []
                for three_pos_index in range(2):
                    if index == 0:
                        ps_heigh_low_px = heigh_px_list.min() + int((three_pos_index + 1)*ps_width/4)
                        self.three_position.append(ps_heigh_low_px)
                    else:
                        ps_heigh_high_px =heigh_px_list.max() - int((three_pos_index + 1)*ps_width/4)
                        self.three_position.append(ps_heigh_high_px)
                self.ratio_list.append(self.get_ps_width(0))
            collision, biger_side = self.head_tail_size_comparision()
            if collision:
                print(self.ratio_list)
                if biger_side == 0:
                    draw_point = np.take(self.ps_pixels, np.where(self.ps_pixels[:,0]==ps_heigh_low_px)[0],0)
                else:
                    draw_point = np.take(self.ps_pixels, np.where(self.ps_pixels[:,0]==ps_heigh_high_px)[0],0)
        return collision, draw_point

    def head_tail_size_comparision(self):
        first_size, second_side = self.ratio_list
        big_side = 0
        if first_size > second_side:
            two_side_ratio = first_size/second_side
        else:
            two_side_ratio = second_side/first_size
            big_side = 1
        # print(two_side_ratio)
        if two_side_ratio > self.config.head_tail_diff_ratio:
            return True, big_side
        else:
            return False, big_side

    def get_ps_width(self, element_index):
        string_width = 0
        for position in self.three_position:
            string_width = string_width + len(np.where(self.ps_pixels[:,element_index] == position)[0])
        return int(string_width/len(self.three_position))

# import sys
# import time
# sys.path.insert(1, r'/home/kai/Documents/minghong/mh_detect_string_collision')
# from config import read_config
# config = read_config.Config([1916,1170])
# start_time = time.time()
# test_class = PS_Check(string_pixel_list, config)
# test_class.get_side_size()
# end_time = time.time()
# print(end_time - start_time)