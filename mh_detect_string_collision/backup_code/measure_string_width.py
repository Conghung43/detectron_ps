

from itertools import count


def measure_string_width(gray_xaxis_list, average_threadhold):
    count = 0
    string_width_list = []
    #Debug
    string_no = 1
    #
    for gray_value in gray_xaxis_list:
        if gray_value > average_threadhold:
            count += 1
        elif count > 0:
            string_width_list.append(count)
            # Debug
            # print(string_no, count)
            string_no += 1
            count = 0
            #
    return string_width_list
