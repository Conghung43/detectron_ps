

def append_new_string(string_property,peak_pixel, string_width):
    string_property.append(string_class.WhiteString())
    string_property[-1].set(peak_pixel,
                            string_width)
    return string_property

string_width_list = measure_string_width.measure_string_width(data_list, 
                                        average_threadhold)
# Based on tracking
temp_array = [len(peak_pixel),len(string_width_list)]
len_loop = temp_array[np.argsort(temp_array)[0]]
if len(string_property) == 0:
    for index in range(len_loop):
        string_property = append_new_string(string_property, 
                                (peak_pixel[index],pattent_xaxis),
                                string_width_list[index])
        # string_property.append(string_class.WhiteString())
        # string_property[-1].set((peak_pixel[index],pattent_xaxis),
        #                         string_width_list[index])
else:
    for index in range(len_loop):
        if index >= len(string_property):
            string_property = append_new_string(string_property, 
                                    (peak_pixel[index],pattent_xaxis),
                                    string_width_list[index])
            continue
        current_index_distance = abs(peak_pixel[index]-string_property[index].get(-1)[0][0])
        if index > 0: 
            previous_index_distance = abs(peak_pixel[index]-string_property[index-1].get(-1)[0][0])
        else: 
            previous_index_distance = config.default_value
        if index < len(string_property)-1:
            next_index_distance = abs(peak_pixel[index]-string_property[index+1].get(-1)[0][0])
        else:
            next_index_distance = config.default_value

        distance_list = [previous_index_distance, current_index_distance, next_index_distance]
        sorted_index = np.argsort(distance_list)
        target_index = index - (sorted_index[0] - 1)
        if target_index >= len(string_property):
            string_property = append_new_string(string_property, 
                                    (peak_pixel[index],pattent_xaxis),
                                    string_width_list[index])
            continue
        lastest_point_position = string_property[target_index].get(-1)[0]
        new_point_position = (peak_pixel[index],pattent_xaxis)
        #Check new point position by angle
        if new_point_verify.angle_verify(lastest_point_position, new_point_position, config.valid_angle):
            string_property[target_index].set(new_point_position, 
                                        string_width_list[index])