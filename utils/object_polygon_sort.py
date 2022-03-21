import numpy as np

def main(object_polygon_list):
    sortable_dictionary = {}
    for object_polygon in object_polygon_list:
        key= get_polygon_key(object_polygon)
        sortable_dictionary[key] = object_polygon
    return [sortable_dictionary[key] for key in sorted (sortable_dictionary.keys())]

def get_polygon_key(object_polygon):
    x_min = object_polygon[:,0].min()
    index = np.where(object_polygon[:,0] == x_min)[0][0]
    return object_polygon[index][1]
