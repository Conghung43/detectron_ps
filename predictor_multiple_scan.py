# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
from matplotlib.transforms import Bbox
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import cam_connect.setup_camera as cam_config
import tqdm

import socket
import time
import utils.geometry_calculation as cal
import numpy as np
import best_fit_rectangle as find_rectangle
import kuka_communicate as kuka

from threading import Thread
import math
import angle_error as err

# cam_pos = [986, 57, 920]
# cam_pos = [773.1, 95.2, 970]
# cam_pos = [772.38, 101.55, 1006.12]
# cam_pos = [787.38, 67.55, 1006.12]
frame =1
depth = 1
point_cloud =1
cam_pos = [770.88, 60.2, 955]
rz = -0.5398864372569108
rx = 0.516833292613569
ry = 0.3720547606821995

err_cubic_equation_x = [0.022708610638252785,5.1334555556640466e-05,1.0503393384805367e-06,-6.423835400510552e-10,-1.0184007807545413e-11,-0.24942819041849262]
err_cubic_equation_y = [0.020736228769246492,-6.27453835869017e-05,1.1822571603782524e-06,7.368353969453356e-10,-1.9115386336165942e-11,0.3189486671654516]

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        
        # self.data = cam.data
        d = "train"
        # DatasetCatalog.register("box_" + d, lambda d=d: get_object_dicts("box/" + d))
        DatasetCatalog.register("pipe", self.fake_func)

        MetadataCatalog.get("pipe_" + d).thing_classes = ['pipe']
        self.metadata = MetadataCatalog.get("pipe")

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

        Thread(target=self.read_cam, args=()).start()

        self.kuka_connect = kuka.connect_kuka()

        self.kuka_connect.sock.recv(1024)
        while True:
            # print('wait for data comming....')
            # received = str(self.sock.recv(1024), "utf-8")
            # received = int(received[:1])
            # # received = 2
            # print(received)
            # if received == 1:
            #     print('handle conveyer then response')
            #     self.send_binary([1])
            #     time.sleep(1)
            # elif received == 2:
            #     try:
            #         self.process_predictions()
            #     except Exception as ex:
            #         print('unexpect exception: ', ex)

            received = str(self.kuka_connect.sock.recv(1024), "utf-8")
            received = int(received[:1])
            
            point_pos = None
            while not point_pos:
                try:
                    point_pos, angle = self.process_predictions()
                    print(point_pos)
                except:
                    print('No pipe in working area')
                    time.sleep(2)
            self.kuka_connect.send_binary([[point_pos[0],point_pos[1], point_pos[2], angle,0,180]])
    def fake_func(self):
        return {}

    def read_cam(self):
        global frame, depth, point_cloud
        cam = cam_config.setup_cam()
        
        # for frame, depth, point_cloud in cam._frame_depth_from_video():
        while True:
            frame, depth, point_cloud = cam.single_data()
            # cv2.imshow('camera', frame)
            # cv2.waitKey(1)

    def bbox2center(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def bbox2angle(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def position_safety_confirm(self, pos):
        x,y,z = pos
        if x < 687 or x > 912:
            return False
        if y < -174 or y > 180:
            return False
        if z < 232 or z > 560:
            return False
        return True

    def get_world_posotion(point_pos, cam_pos):
        P = [278.4875,  90.0875, 692.675 ]
        point_pos = cal.error_compensatiton(point_pos)
        point_world_pos = [cam_pos[0]-point_pos[1],cam_pos[1]-point_pos[0],cam_pos[2]-point_pos[2]]
        return point_world_pos

    def process_predictions(self):
        global frame, depth, point_cloud
        try:
            print('--------------------------------start query data from cam------------------------')
            if cal.get_circle_center(frame):
                h,w,_ = frame.shape
                left_top = [int(w*0.8),int(h*0.5)]
                right_button = [int(w),int(h*0.8)]
                frame = cv2.circle(frame, tuple(left_top), 3, (0, 255, 0), -1)
                frame = cv2.circle(frame, tuple(right_button), 3, (0, 255, 0), -1)
                cv2.imshow('image', frame)
                cv2.waitKey(1)
                print('get_circle')
                return None, None
            # frame, depth, point_cloud = self.cam.single_data()
            predictions = self.predictor(frame)
            video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            predictions = predictions["instances"].to(self.cpu_device)
            vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            boxes = predictions.pred_boxes.tensor.numpy()

            if predictions.has("pred_masks"):
                masks = predictions.pred_masks
            frame_visualizer = Visualizer(frame, self.metadata)
            mask_layers = frame_visualizer._convert_masks(masks)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            for index, mask_layer in enumerate(mask_layers):
                mask_polygon = mask_layer.polygons[0]
                mask_polygon = np.reshape(mask_polygon,(-1,2)).astype(int)

                poly_pc = []
                for poly_point in mask_polygon:
                    poly_pc.append(point_cloud[poly_point[1]][poly_point[0]])
                bbox = boxes[index]
                # for bbox in boxes:
                # Get object center
                x,y= self.bbox2center(bbox)
                points_value = (np.array(poly_pc)*1000).astype(int)
                angle, _ = find_rectangle.get_object_rotation(points_value)
                angle = angle + rz
                _, corner_points = find_rectangle.get_object_rotation(mask_polygon)
                corner_points = np.array(corner_points)
                
                #draw direction vector
                point_1, point_2 = ((corner_points[1]+corner_points[2])/2).astype(int), ((corner_points[3]+corner_points[0])/2).astype(int)
                vis_frame = cv2.line(vis_frame,tuple(point_1),tuple(point_2),(0, 0, 0),1)

                #Temp
                pc_line_list = []
                line_list_2d = cal.bresenham(point_1, point_2)
                redundant_area = int(len(line_list_2d)/ 4)
                line_list_2d = line_list_2d[redundant_area:len(line_list_2d) - redundant_area]
                pre_pc = point_cloud
                for i in range(3):
                    while True:
                        if pre_pc is not point_cloud:
                            break
                    for line_point_2d in line_list_2d:
                        pc_line_list.append(point_cloud[line_point_2d[1]][line_point_2d[0]])
                    # time.sleep(0.1)
                # for i in range(3):
                #     pre_pc = point_cloud
                #     time.sleep(1)
                #     if pre_pc is point_cloud:
                #         print('equal')
                #     else:
                #         print('not equal')

                point_pos = ((np.mean(pc_line_list, axis=0))*1000).astype(int)
                print(len(line_list_2d))


                # point_pos = ((np.mean([point_cloud[y][x],point_cloud[y+2][x],point_cloud[y][x+2]], axis=0))*1000).astype(int)

                # print('===========================================> Angle =', angle)

                vis_frame = cv2.circle(vis_frame, (int(x),int(y)), 3, (0, 255, 0), -1)
                cv2.imshow('image', vis_frame)
                cv2.waitKey(1)
                

                #Rotation handling
                point_pos = ((point_pos*cal.Rz(math.radians(rz))*cal.Rx(math.radians(rx))*cal.Ry(math.radians(ry)))[0].tolist())[0]
                #Camera error handling
                point_pos[0] = point_pos[0] - err.objective_5(point_pos[0], err_cubic_equation_x[0],
                                                                                            err_cubic_equation_x[1], err_cubic_equation_x[2],
                                                                                            err_cubic_equation_x[3], err_cubic_equation_x[4],
                                                                                            err_cubic_equation_x[5])

                point_pos[1] = point_pos[1] - err.objective_5(point_pos[1], err_cubic_equation_y[0],
                                                                                            err_cubic_equation_y[1], err_cubic_equation_y[2],
                                                                                            err_cubic_equation_y[3], err_cubic_equation_y[4],
                                                                                            err_cubic_equation_y[5])
                print(point_pos)
                # put your code here
                # cv2.destroyAllWindows()
                # point_pos = np.array(point_pos)
                point_pos = cal.get_world_position(point_pos, cam_pos)
                # point_pos[2] = point_pos[2] - 28
                print('Z position based camera: ', point_pos[2])
                # point_pos[2] = 235.5
                # point_pos[2] = 260
                print(point_pos)
                if not self.position_safety_confirm(point_pos):
                    print('--------->position error')
                    continue
                
                send_point = point_pos
                send_angle = angle
            # time.sleep(20)
            # print('-----------------run------------')
            return send_point, send_angle
        except Exception as ex:
            print(ex)
            return None, None
            # if cv2.waitKey(1) == 27:
            #     break  # esc to quit
        # return vis_frame
