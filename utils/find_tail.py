import math
import numpy as np
import utils.geometry_calculation as cal

def distance_with_vector_calculation(p, v, vector, d, direction):
    calculate_value = v*math.sqrt(pow(d,2)/(pow(vector[0],2) + pow(vector[1],2) + pow(vector[2],2)))
    if direction:
        return_value = p + calculate_value
    else:
        return_value = p - calculate_value
    return return_value

def get_point_from_distance_with_2point_3d(corner_point,edge_point, distance, outside):
    direction_vector = np.array(corner_point) - np.array(edge_point)
    return_point = []
    for index, element in enumerate(corner_point):
        value = distance_with_vector_calculation(element, direction_vector[index], direction_vector, distance, outside)
        return_point.append(value)
    
    return return_point

# print(get_point_from_distance_with_2point_3d([4,0,0], [0,0,0], 2, True))

def get_direction_line_equation(rec_corner):
    point1 = (np.array(rec_corner[0]) +np.array(rec_corner[-1]))/2
    point2 = (np.array(rec_corner[1]) +np.array(rec_corner[-2]))/2
    line_equation = cal.get_distance_point_line_3d

def find_narrow(rec_corners):
    crop_size = cal.get_distance_two_point_2d(rec_corners[0],rec_corners[1])/8
    repick_rec_corners = []
    for i in range(0,len(rec_corners),2):
        a = get_point_from_distance_with_2point_3d([rec_corners[i][0], rec_corners[i][1], 0], 
                                                    [rec_corners[i+1][0], rec_corners[i+1][1], 0], crop_size, False)
        b = get_point_from_distance_with_2point_3d([rec_corners[i+1][0], rec_corners[i+1][1], 0], 
                                                    [rec_corners[i][0], rec_corners[i][1], 0], crop_size, False)
        repick_rec_corners.append(a[:2])
        repick_rec_corners.append(b[:2])
    return repick_rec_corners


print(find_narrow([[0,0],[0,4],[4,2],[0,2]]))