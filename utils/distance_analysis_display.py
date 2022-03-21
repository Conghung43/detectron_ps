import cv2
import numpy as np
def main(vis_frame,list_error_pos):
    for error_pos in list_error_pos:
        if len(error_pos) == 2:
            if error_pos[1] < 30:
                error_pos[1] = 30
            # start_point = np.clip((np.array(error_pos[0]) - np.array([error_pos[1]/2,error_pos[1]/2])).astype(int),0,vis_frame.shape[1])
            # end_point = np.clip((np.array(error_pos[0]) + np.array([error_pos[1]/2,error_pos[1]/2])).astype(int),0,vis_frame.shape[1])
            start_point = (np.array(error_pos[0]) - np.array([error_pos[1]/2,error_pos[1]/2])).astype(int)
            end_point = (np.array(error_pos[0]) + np.array([error_pos[1]/2,error_pos[1]/2])).astype(int)
            vis_frame = cv2.rectangle(vis_frame, tuple(start_point), tuple(end_point), [0,0,255], 5)
        else:
            for pos_index in range(len(error_pos)):
                if pos_index == len(error_pos)-1:
                    vis_frame = cv2.line(vis_frame, 
                                        tuple(error_pos[pos_index]), 
                                        tuple(error_pos[0]), 
                                        [0,0,255], 5)
                else:
                    vis_frame = cv2.line(vis_frame, 
                                        tuple(error_pos[pos_index]), 
                                        tuple(error_pos[pos_index+1]), 
                                        [0,0,255], 5)