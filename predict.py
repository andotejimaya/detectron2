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

    basepath = os.path.dirname(__file__)

    now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    result_dir = "/tmp"
    result_imagepath = os.path.join(result_dir, os.path.basename(imagepath).replace(".jpg",".result.jpg"))
    result_imagepath2 = result_imagepath.replace(".result.jpg",".result2.jpg")
    result_jsonpath = result_imagepath.replace(".result.jpg",".json")

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
    metadata.thing_classes = ["room"]

    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(result_imagepath, out.get_image()[:, :, ::-1])

#    for box in outputs["instances"].pred_boxes.to('cpu'):
#        visualizer.draw_box(box)
#        box = (round(box[0]), round(box[1]), round(box[2]) - round(box[0]), round(box[3] - box[1]))
 #       out = v.draw_text(f"{box[2:4]}", (box[0], box[1]))
#    for mask in outputs["instances"].pred_masks.cpu().numpy()
    visualizer = Visualizer(im[:, :, ::-1], metadata=None, scale=1.0)
    for mask in outputs["instances"].pred_masks.to('cpu'):        
#        visualizer.draw_binary_mask(mask.numpy(), color=None, edge_color=None, text=None)
        visualizer.draw_soft_mask(mask.numpy())

    out = visualizer.get_output() 

    cv2.imwrite(result_imagepath2, out.get_image()[:, :, ::-1])

    inputs = [{'image_id': 0}]
    livs = LVISEvaluator("madori")
    livs.reset()
    livs.process(inputs, [outputs])

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

