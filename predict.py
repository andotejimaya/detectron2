#import detectron2
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation import LVISEvaluator

import argparse
import cv2
import os
import json
import datetime

def main (imagepath):
    #os.environ['MPLCONFIGDIR'] = "/tmp/"

    basepath = os.path.dirname(__file__)

    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    #result_dir = os.path.join(os.getcwd(), now)
    #result_dir = os.path.join("/tmp", now)
    result_dir = "/tmp"
    #print (os.path.exists(result_dir))
    if not os.path.exists(result_dir):
        aaa = os.makedirs(result_dir, exist_ok=True)
        #os.mkdir(result_dir)
        #print ('os.makedirs',aaa)


    result_imagepath = os.path.join(result_dir, os.path.basename(imagepath).replace(".jpg",".result.jpg"))
    result_jsonpath = result_imagepath.replace(".result.jpg",".json")
    #print(result_imagepath)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 #
    cfg.MODEL.WEIGHTS = os.path.join(basepath, cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # ?
    cfg.MODEL.DEVICE = "cpu" # ?

    predictor = DefaultPredictor(cfg)

    im = cv2.imread(imagepath)

    outputs = predictor(im)
    instances = outputs['instances'].to("cpu")

    register_coco_instances("madori", {}, os.path.join(basepath, "../data/dataset/annotations/annotations.json"), os.path.join(basepath,"../data/dataset/train/"))
    metadata = MetadataCatalog.get("madori")
    #metadata.thing_classes = ["floor","room","toilet","bathroom"]
    #metadata.thing_classes = ["room","toilet","bathroom"]
    metadata.thing_classes = ["room"]
    #metadata.thing_classes = ["_background_","room"]

    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(result_imagepath, v.get_image()[:, :, ::-1])

    #print("instances={0}".format(len(instances)))
    #print("pred_boxes={0}".format(len(instances.pred_boxes)))
    #print("pred_masks={0}".format(len(instances.pred_masks)))
    #print("pred_classes={0}".format(len(instances.pred_classes)))
    #print("scores={0}".format(len(instances.scores)))

    pred_class = instances.pred_classes[0]
    #print(pred_class)

    outputs2 = []
    outputs2.append(outputs)

    inputs = []
    aaa = {}
    aaa["image_id"] = 0
    inputs.append(aaa)

    livs = LVISEvaluator("madori")
    livs.reset()
    livs.process(inputs, outputs2)

    # add tag
    for i in range(len(livs._predictions[0]['instances'])):
        livs._predictions[0]['instances'][i]['tag'] = '{:05}'.format(i)

    with open(result_jsonpath, mode="w") as f:
        f.write(json.dumps(livs._predictions))

    return result_jsonpath, result_imagepath

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', type=str, required=True)
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    main(args.imagepath)

