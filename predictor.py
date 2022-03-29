from tkinter.messagebox import NO
import cv2
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import utils.geometry_calculation as cal
import numpy as np
from utils import distance_analysis_tensor as distance_analysis
from utils import distance_analysis_display
from utils import object_polygon_sort
from config import read_config
import time
from display import draw_polygon
from auto_labeling import polygons_adjustment
import test_function
from auto_labeling import generate_json
class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        d = "train"
        DatasetCatalog.register("pipe", self.fake_func)
        MetadataCatalog.get("pipe_" + d).thing_classes = ['pipe']
        self.metadata = MetadataCatalog.get("pipe")
        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        self.process_predictions()
        # self.test()
    def fake_func(self):
        return {}
    def test(self):
        path = f"/home/kai/Documents/DATASET/minghong/ps_data/image_0328/IMG_1621/"
        json_data = {}
        video = cv2.VideoCapture(f'/home/kai/Documents/DATASET/minghong/ps_data/video/IMG_1621.MOV')
        jump_range = 10
        frame_count = 0
        while True:
            # frame = cv2.imread('dummy_data/objects_collection.jpg')
            ret, frame = video.read()
            frame_count += 1
            if not ret:
                break
            if frame_count % jump_range != 0:
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h,w,_ = frame.shape
            frame = frame[int(h*5/16):int(h*13/16), int(w/2):w]
            frame_visualizer = Visualizer(frame, self.metadata)
            masks = test_function.create_masks(frame)
            mask_layers = frame_visualizer._convert_masks(masks)
            print('mask_layers =', len(mask_layers))
            object_polygon_list =[]
            
            for object_polygon in mask_layers:
                len_polygon = len(object_polygon.polygons)
                polygon_reshape = None
                if len_polygon == 1:
                    polygon_reshape = np.reshape(object_polygon.polygons[0],(-1,2)).astype(int)
                    
                    #calculate volume of ps
                    polygon_reshape = polygons_adjustment.change_polygon_point(polygon_reshape)
                    # object_polygon_list.append(polygon_reshape)
                    # mask_index = np.where(object_polygon.mask == 1)
                    # len_mask_index = len(mask_index[0])
                    # lenth_mask_index = mask_index[1].max() - mask_index[1].min()
                    # print(len_mask_index, lenth_mask_index, len_mask_index/lenth_mask_index)
                elif len_polygon > 1:
                    polygon_reshape = polygons_adjustment.get_main_polygon(object_polygon.polygons).reshape(-1, 2)
                    polygon_reshape = polygons_adjustment.change_polygon_point(polygon_reshape)
                    # object_polygon_list.append(polygon_reshape)
                else:
                    continue
                polygon_reshape = polygons_adjustment.get_sorter_polygon(polygon_reshape)
                if polygon_reshape is None:
                    continue
                object_polygon_list.append(np.array(polygon_reshape))
            sorted_object_polygon = object_polygon_sort.main(object_polygon_list)
            # f =  open('dummy_data/auto_detect_polygon.npy', 'wb') 
            # np.save(f, [sorted_object_polygon[4]])
            # f.close()
            # for object_polygon in sorted_object_polygon:
            # draw_polygon.main(sorted_object_polygon, frame)
            key, value = generate_json.save_image_gen_json(
                frame, 
                sorted_object_polygon,
                path)
            if value is None:
                continue
            json_data[key] = value
        generate_json.write_json(json_data,path)


    def bbox2center(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def bbox2angle(self, bbox):
        l, t, r, b = bbox
        return int((l+r)/2), int((t+b)/2)

    def process_predictions(self):
        
        #Read cam, video
        video_name = 'IMG_1621'
        path = f"/home/kai/Documents/DATASET/minghong/ps_data/{video_name}/"
        cap = cv2.VideoCapture(f'/home/kai/Documents/DATASET/minghong/ps_data/video/{video_name}.MOV')
        config = read_config.Config([cap.get(4), cap.get(3)])
        # video_export = cv2.VideoWriter(f"video_log/video_{time.time()}.mp4", 0x7634706d, 10, (540,960))#(int(cap.get(4)),int(cap.get(3))
        frame_count = 0
        while True:
            # try:
                # print('--------------------------------start query data from cam------------------------')
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                # frame_count += 1
                # if frame_count % 50 !=0:
                #     continue
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                h,w,_ = frame.shape
                frame = frame[int(h*5/16):int(h*13/16), int(w/2):w]
                # cv2.imwrite(f'/home/kai/Documents/DATASET/minghong/ps_data/image_0303/{video_name}/{time.time()}.jpg', frame)
                # continue
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = self.predictor(frame)
                video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = predictions["instances"].to(self.cpu_device)
                # Comment this method because of causing large time consumtion
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                cv2.imwrite('image.jpg', vis_frame.get_image())
                continue
                # vis_frame = frame
                # boxes = predictions.pred_boxes.tensor.numpy()
                if predictions.has("pred_masks"):
                    masks = predictions.pred_masks
                frame_visualizer = Visualizer(frame, self.metadata)
                mask_layers = frame_visualizer._convert_masks(masks)
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
                cv2.imwrite('image.jpg', vis_frame)

                print('mask_layers =', len(mask_layers))
                object_polygon_list =[]
                for object_polygon in mask_layers:
                    len_polygon = len(object_polygon.polygons)
                    if len_polygon == 1:
                        polygon_reshape = np.reshape(object_polygon.polygons[0],(-1,2)).astype(int)
                        object_polygon_list.append(polygon_reshape)
                    elif len_polygon > 1:
                        polygon_reshape = polygons_adjustment.get_main_polygon(object_polygon.polygons).reshape(-1, 2)
                        object_polygon_list.append(polygon_reshape)
                sorted_object_polygon = object_polygon_sort.main(object_polygon_list)

                # for object_polygon in sorted_object_polygon:
                draw_polygon.main(sorted_object_polygon, frame)

                list_error_pos = distance_analysis.main(sorted_object_polygon, config.shortest_distance)
                distance_analysis_display.main(vis_frame, list_error_pos)
                
                # video_export.write(vis_frame)
        # video_export.release()
                # time.sleep(1)
            # except Exception as ex:
            #     print(ex)
                    # if cv2.waitKey(1) == 27:
                    #     break  # esc to quit
                # return vis_frame
