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
    def fake_func(self):
        return {}

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
                predictions = self.predictor(frame)
                video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = predictions["instances"].to(self.cpu_device)
                # Comment this method because of causing large time consumtion
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                # vis_frame = frame
                # boxes = predictions.pred_boxes.tensor.numpy()
                if predictions.has("pred_masks"):
                    masks = predictions.pred_masks
                frame_visualizer = Visualizer(frame, self.metadata)
                mask_layers = frame_visualizer._convert_masks(masks)
                # Converts Matplotlib RGB format to OpenCV BGR format
                for mask_layer in mask_layers:
                    print(mask_layer.area())
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
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
                # draw_polygon.main(sorted_object_polygon, frame)

                list_error_pos = distance_analysis.main(sorted_object_polygon, config.shortest_distance)
                distance_analysis_display.main(vis_frame, list_error_pos)
                cv2.imwrite('image.jpg', vis_frame)
                
                # video_export.write(vis_frame)
        # video_export.release()
                # time.sleep(1)
            # except Exception as ex:
            #     print(ex)
                    # if cv2.waitKey(1) == 27:
                    #     break  # esc to quit
                # return vis_frame
