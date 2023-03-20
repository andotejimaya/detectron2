#import detectron2
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import json

basepath = os.path.dirname(__file__)

json_path = "../data/dataset/annotations/annotations.json"
image_path = "../data/dataset/train"

with open(json_path, 'r') as f:
    anno = json.load(f)

classes = len(anno['categories'])


register_coco_instances("madori", {}, json_path, image_path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("madori",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # default: 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = classes

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()