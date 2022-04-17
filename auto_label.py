import cv2
import torch
import argparse
import multiprocessing as mp
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import time
import utils.geometry_calculation as cal
import numpy as np
from auto_labeling import generate_json
from utils import object_polygon_sort
from config import read_config
import time
from auto_labeling import polygons_adjustment
import test_function
import os
video_name = 'IMG_1625'
path = f"/home/kai/Documents/DATASET/minghong/ps_data/{video_name}/"
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
        # self.based_prediction()
        self.based_hsv()

    def fake_func(self):
        return {}

    def create_folder(self, path):
        try:
            os.makedirs(path)
        except:
            pass

    def based_hsv(self):
        path = f"/home/kai/Documents/DATASET/mh/ps/image_0417/IMG_1621/"
        self.create_folder(path)
        json_data = {}
        video = cv2.VideoCapture(f'/home/kai/Documents/DATASET/mh/ps/video/IMG_1621.MOV')
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

    def based_prediction(self):
        global frame, depth, point_cloud
        # frame = cv2.imread('/home/kai/Documents/DATASET/minghong/ps_data/video/IMG_1540.MOV')
        video_name = 'IMG_154'
        for index in range(0,8,1):
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

            generate_json.write_json(json_data,path)
# constants
WINDOW_NAME = "COCO detections"
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', '/home/kai/Documents/DATASET/mh/ps/weight/model_final_0328.pth'])#/home/kai/Documents/DATASET/box/image_small_box/model_0005999.pth, 'model_final.pth'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

# Based on detectron2
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
