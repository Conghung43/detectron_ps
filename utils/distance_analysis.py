from scipy.spatial.distance import cdist
import numpy as np
# x1 = x2 = np.linspace(-1000, 1000, 2001)
# y1 = (lambda x, a, b: a*x + b)(x1, 2, 1)
# y2 = (lambda x, a, b: a*(x-2)**2 + b)(x2, 2, 10)

# R1 = np.vstack((x1,y1)).T
# R2 = np.vstack((x2,y2)).T

# R1 = np.matrix([[1,1],[1,1], [2,1]])
# R2 = np.matrix([[1,2],[1,3],[2,3]])
def main(sorted_object_polygon, threadhold_config):
    list_error_pos = []
    for index, object_polygon in enumerate(sorted_object_polygon):
        if index >= len(sorted_object_polygon)-1:
            break
        two_line_distance = get_shortest_distance(object_polygon,sorted_object_polygon[index+1])
        if two_line_distance < threadhold_config:
            # print(two_line_distance)
            error_pos = find_error_position(object_polygon, sorted_object_polygon[index+1], threadhold_config)
            if error_pos is not None:
                list_error_pos.append(error_pos)
    return list_error_pos
            # f =  open('dummy_data/{}.npy'.format('object_polygon'), 'wb') 
            # np.save(f, object_polygon)
            # np.save(f,  sorted_object_polygon[index+1])
            # f.close()

def get_shortest_distance(R1,R2):
    dists = cdist(R1,R2)
    return dists.min()

def find_error_position(R1, R2, threadhold_config):
    r1_min = R1[np.where(R1[:,0] == R1[:,0].min())][0]
    r1_max = R1[np.where(R1[:,0] == R1[:,0].max())][0]
    r2_min = R2[np.where(R2[:,0] == R2[:,0].min())][0]
    r2_max = R2[np.where(R2[:,0] == R2[:,0].max())][0]
    if r1_min[0] > r2_min[0]:
        head_distance = cdist([r1_min],R2).min()
        r_min = r1_min
    else:
        head_distance = cdist([r2_min],R1).min()
        r_min = r2_min
    if r1_max[0] < r2_max[0]:
        tail_distance = cdist([r1_max],R2).min()
        r_max = r1_max
    else:
        tail_distance = cdist([r2_max],R1).min()
        r_max = r2_max
    if head_distance < threadhold_config and tail_distance < threadhold_config:
        return None
    else:
        if head_distance < threadhold_config and tail_distance > threadhold_config:
            return [r_min, tail_distance]
        elif head_distance > threadhold_config and tail_distance < threadhold_config:
            return [r_max, head_distance]
        else:
            return [r1_min, r1_max, r2_max, r2_min]

# f =  open('dummy_data/{}.npy'.format('object_polygon'), 'rb')
# R1 = np.load(f, allow_pickle=True)
# R2 = np.load(f, allow_pickle=True)
# find_error_position(R1, R2, 20)