import cv2
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import time
import utils.geometry_calculation as cal
import numpy as np
import auto_labeling.polygons_adjustment as polygons_adjustment
from auto_labeling import generate_json

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
        # Thread(target=self.read_cam, args=()).start()
        # try:
        self.process_predictions()
            # print(point_pos)
        # except:
        #     print('No pipe in working area')
        #     time.sleep(2)
    def fake_func(self):
        return {}

    def bbox2center(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def bbox2angle(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def get_world_posotion(point_pos, cam_pos):
        P = [278.4875,  90.0875, 692.675 ]
        point_pos = cal.error_compensatiton(point_pos)
        point_world_pos = [cam_pos[0]-point_pos[1],cam_pos[1]-point_pos[0],cam_pos[2]-point_pos[2]]
        return point_world_pos

    def process_predictions(self):
        global frame, depth, point_cloud
        # frame = cv2.imread('/home/kai/Documents/DATASET/minghong/ps_data/video/IMG_1540.MOV')
        video_name = 'IMG_154'
        for index in range(0,8,1):
            # continue
            # path = f"/home/kai/Documents/DATASET/minghong/ps_data/image_0303/{video_name}{index}/"
            # cap = cv2.VideoCapture(f'/home/kai/Documents/DATASET/minghong/ps_data/video/{video_name}{index}.MOV')
            video_name = 'IMG_1625'
            path = f"/home/kai/Documents/DATASET/minghong/ps_data/{video_name}/"
            cap = cv2.VideoCapture(f'/home/kai/Documents/DATASET/minghong/ps_data/video/{video_name}.MOV')
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            json_data = {}
            jump_range = 10
            frame_count = 0
            while True:
                # try:
                    start_time = time.time()
                    frame_count += 1
                    print('--------------------------------start query data from cam------------------------')
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % jump_range != 0:
                        continue
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
                    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
                    # cv2.imwrite('image.jpg', vis_frame)
                    # cv2.waitKey(1)
                    # continue
                    # f =  open('dummy_data/{}.npy'.format('mask_polygon'), 'a') 
                    # np.save('dummy_data/{}.npy'.format('mask_polygon'), mask_layers)
                    # f.close()
                    objects_polygons = []
                    for index, mask_layer in enumerate(mask_layers):
                        if len(mask_layer.polygons) == 0:
                            continue
                        mask_polygon = mask_layer.polygons[0]
                        mask_polygon = np.reshape(mask_polygon,(-1,2)).astype(int)
                        polygons = polygons_adjustment.get_sorter_polygon(mask_polygon)

                        for index, point in enumerate(polygons):
                            if index >= len(polygons) - 1:
                                vis_frame = cv2.line(vis_frame,tuple(point),tuple(polygons[0]),(0, 255, 0),1)
                            else:
                                vis_frame = cv2.line(vis_frame,tuple(point),tuple(polygons[index + 1]),(0, 255, 0),1)

                        if polygons is not None:
                            objects_polygons.append(polygons)
                    key, value = generate_json.save_image_gen_json(vis_frame, objects_polygons,path)
                    if value is None:
                        continue
                    json_data[key] = value
                    end_time = time.time()
                    ellapsed_time = end_time - start_time
                    # print(f'Remaining time for this video = {(total_frame*ellapsed_time/jump_range) - ((total_frame-frame_count)/total_frame*ellapsed_time)}')
                # except Exception as ex:
                #     print(ex)
            generate_json.write_json(json_data,path)
            #     for index, point in enumerate(polygons):
            #         if index >= len(polygons) - 1:
            #             vis_frame = cv2.line(vis_frame,tuple(point),tuple(polygons[0]),(0, 255, 0),1)
            #         else:
            #             vis_frame = cv2.line(vis_frame,tuple(point),tuple(polygons[index + 1]),(0, 255, 0),1)
            #     cv2.imwrite('image.jpg', vis_frame)
            #     continue
            #     poly_pc = []
            #     for poly_point in mask_polygon:
            #         poly_pc.append(point_cloud[poly_point[1]][poly_point[0]])
            #     bbox = boxes[index]
            #     # for bbox in boxes:
            #     # Get object center
            #     x,y= self.bbox2center(bbox)
            #     points_value = (np.array(poly_pc)*1000).astype(int)
            #     angle, rec_side = find_rectangle.get_object_rotation(points_value)
            #     angle = angle + rz
            #     _, corner_points = find_rectangle.get_object_rotation(mask_polygon)
            #     corner_points = np.array(corner_points)
                
            #     #draw direction vector
            #     point_1, point_2 = ((corner_points[1]+corner_points[2])/2).astype(int), ((corner_points[3]+corner_points[0])/2).astype(int)
            #     vis_frame = cv2.line(vis_frame,tuple(point_1),tuple(point_2),(0, 0, 0),1)

            #     #Temp
            #     pc_line_list = []
            #     line_list_2d = cal.bresenham(point_1, point_2)
            #     redundant_area = int(len(line_list_2d)/ 6)
            #     line_list_2d = line_list_2d[redundant_area:len(line_list_2d) - redundant_area]
            #     # loop_count = 0
            #     # while True:
            #     #     if pre_pc is not point_cloud:
            #     #         loop_count += 1
            #     for line_point_2d in line_list_2d:
            #         pc_line_list.append(point_cloud[line_point_2d[1]][line_point_2d[0]])
            #             # if loop_count >1:
            #             #     break
            #     # mat.data_show_3d(pc_line_list)
                        
            #         # time.sleep(0.1)
                
            #     # for i in range(3):
            #     #     pre_pc = point_cloud
            #     #     time.sleep(1)
            #     #     if pre_pc is point_cloud:
            #     #         print('equal')
            #     #     else:
            #     #         print('not equal')

            #     point_pos = ((np.mean(pc_line_list, axis=0))*1000).astype(int)
            #     print(len(line_list_2d))


            #     # point_pos = ((np.mean([point_cloud[y][x],point_cloud[y+2][x],point_cloud[y][x+2]], axis=0))*1000).astype(int)

            #     # print('===========================================> Angle =', angle)

            #     vis_frame = cv2.circle(vis_frame, (int(x),int(y)), 3, (0, 255, 0), -1)
            #     cv2.imshow('image', vis_frame)
            #     cv2.waitKey(1)
                

            #     #Rotation handling
            #     point_pos = ((point_pos*cal.Rz(math.radians(rz))*cal.Rx(math.radians(rx))*cal.Ry(math.radians(ry)))[0].tolist())[0]
            #     #Camera error handling
            #     point_pos[0] = point_pos[0] - err.objective_5(point_pos[0], err_cubic_equation_x[0],
            #                                                                                 err_cubic_equation_x[1], err_cubic_equation_x[2],
            #                                                                                 err_cubic_equation_x[3], err_cubic_equation_x[4],
            #                                                                                 err_cubic_equation_x[5])

            #     point_pos[1] = point_pos[1] - err.objective_5(point_pos[1], err_cubic_equation_y[0],
            #                                                                                 err_cubic_equation_y[1], err_cubic_equation_y[2],
            #                                                                                 err_cubic_equation_y[3], err_cubic_equation_y[4],
            #                                                                                 err_cubic_equation_y[5])
            #     print(point_pos)
            #     # put your code here
            #     # cv2.destroyAllWindows()
            #     # point_pos = np.array(point_pos)
            #     point_pos = cal.get_world_position(point_pos, cam_pos)
            #     # point_pos[2] = point_pos[2] - 28
            #     print('Z position based camera: ', point_pos[2])
            #     point_pos[2] = 235.5
            #     # point_pos[2] = 260
            #     print(point_pos)
            #     if not self.position_safety_confirm(point_pos):
            #         print('--------->position error')
            #         continue
                
            #     send_point = point_pos
            #     send_angle = angle
            # # time.sleep(20)
            # # print('-----------------run------------')
            # return send_point, send_angle
        # except Exception as ex:
            # print(ex)
            # return None, None
            # if cv2.waitKey(1) == 27:
            #     break  # esc to quit
        # return vis_frame
