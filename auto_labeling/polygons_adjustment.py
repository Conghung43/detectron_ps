import numpy as np
import utils.fit_skspatial as fit_skspatial

def get_sorter_polygon(mask_polygon):
    #Define number of polygon edges
    separate_part = 14
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
        if x_list.max() in temp_list[:,0]:
            # New algorithm
            max_index_list = np.where(temp_list[:,0] == x_list.max())[0]
            separated_list.append(temp_list[max_index_list[0]])
            separated_list.append(temp_list[max_index_list[-1]])

            #Commit old algorithm
            # total_point_side = np.where(mask_polygon[:,0] == x_list.max())[0]
            # if len(total_point_side)>1:
            #     if x_list.max() != temp_list[-1][0]:
            #         separated_list.append(mask_polygon[total_point_side[0]])
            #     separated_list.append(mask_polygon[total_point_side[-1]])
            # else:
            #     separated_list.append(mask_polygon[total_point_side[0]])

        if x_list.min() in temp_list[:,0]:
            min_index_list = np.where(temp_list[:,0] == x_list.min())[0]
            separated_list.append(temp_list[min_index_list[0]])
            separated_list.append(temp_list[min_index_list[-1]])

            # old algorithm
            # total_point_side = np.where(mask_polygon[:,0] == x_list.min())[0]
            # if len(total_point_side)>1:
            #     if x_list.min() != temp_list[-1][0]:
            #         separated_list.append(mask_polygon[total_point_side[0]])
            #     separated_list.append(mask_polygon[total_point_side[-1]])
            # else:
            #     separated_list.append(mask_polygon[total_point_side[0]])
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

def change_polygon_point(polygons):
    width = 8
    polygons = polygons.astype(int)
    polygons_list = polygons.tolist()
    for index in range(polygons[:,0].min()+1,polygons[:,0].max()-1,1):
        where_index = np.where(polygons[:,0] == index)
        scanned_data = polygons[where_index]
        if len(where_index[0]) == 2:
            where_index = where_index[0].tolist()
            average = (scanned_data[0][1] + scanned_data[1][1])/2
            # if scanned_data[0][1] - scanned_data[1][1] <= 1:
            #     print('debug')
            if scanned_data[0][1] < scanned_data[1][1]:
                polygons_list[where_index[0]][1] = int(average - width/2)
                polygons_list[where_index[1]][1] = int(average + width/2)
            else:
                polygons_list[where_index[0]][1] = int(average + width/2)
                polygons_list[where_index[1]][1] = int(average - width/2 )
        elif len(where_index[0]) > 2:
            average = np.mean(scanned_data,axis = 0)[1]
            where_index = where_index[0].tolist()
            for index, data in enumerate(scanned_data):
                if data[1] > average:
                    polygons_list[where_index[index]][1] = int(average + width/2)
                else:
                    polygons_list[where_index[index]][1] = int(average - width/2)
        else:
            print('debug')
    return np.array(polygons_list)