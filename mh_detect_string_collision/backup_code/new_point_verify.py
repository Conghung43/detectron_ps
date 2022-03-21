import imp
from utils import calculation_method
import numpy as np
def angle_verify(lastest_point, new_point, valid_angle):
    new_vector = np.array(lastest_point) - np.array(new_point)
    angle = calculation_method.get_angle_two_vector(new_vector, (0,1))
    if abs(angle) > 90:
        angle = 180 - abs(angle)
    if angle < valid_angle:
        return True
    else:
        return False