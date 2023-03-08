#import detectron2
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

import cv2
import os
import subprocess

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  #
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # ?
cfg.MODEL.DEVICE = "cpu" # ?

predictor = DefaultPredictor(cfg)

imgPath = "../data/dataset/train/JPEGImages/00007.jpg" #
im = cv2.imread(imgPath)

outputs = predictor(im)
register_coco_instances("madori", {}, "../data/dataset/annotations/train_annotations.json", "../data/dataset/train/")
metadata = MetadataCatalog.get("madori")
v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite("./aaaa.jpg", v.get_image()[:, :, ::-1])
