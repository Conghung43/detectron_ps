import imp
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np

#input: List of x axis pixel color data 
def get_string_property(gray_xaxis_list,threadhold_from_peak, peak_distance, start_left):
    peaks, peak_heigh_dict = find_peaks(gray_xaxis_list, 
                                    height=100, 
                                    distance= peak_distance)
    #Adujst peak height based on peak_heigh_dict
    average_peak_heigh = np.mean(peak_heigh_dict['peak_heights']).astype(int)
    if average_peak_heigh - threadhold_from_peak > 100:
        peaks, peak_heigh_dict = find_peaks(gray_xaxis_list, 
                                    height=average_peak_heigh - threadhold_from_peak, 
                                    distance= peak_distance)
    # average_threadhold_heigh = np.mean(peak_heigh_dict['peak_heights']).astype(int) -threadhold_from_peak
    average_threadhold_heigh = (peak_heigh_dict['peak_heights'].max()).astype(int) -threadhold_from_peak
    return average_threadhold_heigh, peaks + start_left

    # peaks = list(np.asarray(peaks) + start_left-1)

    # count_dict = Counter(data_list)
    # plt.plot(list(range(start_left, end_right)),data_list)
    plt.plot(list(range(0, len(gray_xaxis_list))),gray_xaxis_list)
    plt.plot(peaks, gray_xaxis_list[peaks], "x")
    # plt.plot([1]*len(data_list),data_list)
    # plt.plot(list(count_dict.keys()),list(count_dict.values()))
    # plt.pause(3)
    # plt.clf()
    plt.show()