import numpy as np
import utils.fit_skspatial as fit_skspatial

def get_sorter_polygon(mask_polygon):
    #Define number of polygon edges
    separate_part = 10
    x_list = mask_polygon[:,0]
    # Define how many points between two polygon points
    jump_points = int(len(mask_polygon)/(separate_part*2))
    if jump_points < separate_part:
        return None
    current_position = 0
    separated_list = []
    while True:
        if current_position >= len(mask_polygon):
            # temp_list = mask_polygon[current_position:]
            # separated_list.append(temp_list[-1])
            break
        current_position = current_position + jump_points
        temp_list = mask_polygon[current_position - jump_points:current_position]
        if x_list.max() in temp_list:
            total_point_side = np.where(mask_polygon[:,0] == x_list.max())[0]
            if len(total_point_side)>1:
                if x_list.max() != temp_list[-1][0]:
                    separated_list.append(mask_polygon[total_point_side[0]])
                separated_list.append(mask_polygon[total_point_side[-1]])
            else:
                separated_list.append(mask_polygon[total_point_side[0]])

        if x_list.min() in temp_list:
            total_point_side = np.where(mask_polygon[:,0] == x_list.min())[0]
            if len(total_point_side)>1:
                if x_list.min() != temp_list[-1][0]:
                    separated_list.append(mask_polygon[total_point_side[0]])
                separated_list.append(mask_polygon[total_point_side[-1]])
            else:
                separated_list.append(mask_polygon[total_point_side[0]])
        if len(temp_list) > 0:
            separated_list.append(temp_list[-1])
    return separated_list

# f =  open('dummy_data/{}.npy'.format('mask_polygon'), 'rb')
# mask_polygon = np.load(f, allow_pickle=True)
# for index, mask_layer in enumerate(mask_polygon):
#     mask_polygon = mask_layer.polygons[0]
#     mask_polygon = np.reshape(mask_polygon,(-1,2)).astype(int)
#     get_sorter_polygon(mask_polygon)

def get_main_polygon(polygons):
    temp_len = 0
    temp_index = 0
    for index, polygon in enumerate(polygons):
        len_polygon = len(polygon)
        if len_polygon > temp_len:
            temp_len = len_polygon
            temp_index = index
    return polygons[temp_index]
