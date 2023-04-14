import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from detectron2.structures.boxes import Boxes
import torch
import random
from torchvision.ops import masks_to_boxes
import csv
import numpy as np
import pandas as pd
from PIL import Image

from detectron2.structures.masks import BitMasks
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from alfred.vis.image.mask import (
    label2color_mask,
    vis_bitmasks,
    vis_bitmasks_with_classes,
)
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from alfred.utils.file_io import ImageSourceIter
from yolov7.config import add_yolo_config

import numpy as np
# constants
WINDOW_NAME = "COCO detections"


class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            print("image after transform: ", image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            # predictions, pure_t = self.model([inputs])
            predictions = self.model([inputs])
            
            predictions = predictions[0]
            c = time.time() - tic
            print("cost: {}, fps: {}".format(c, 1 / c))
            return predictions


def setup_cfg(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.YOLO.CONF_THRESHOLD = args.confidence_threshold
    cfg.MODEL.YOLO.NMS_THRESHOLD = args.nms_threshold
    cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1
    # force devices based on user device
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.INPUT.MAX_SIZE_TEST = 600  # 90ms
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument(
        '-i',
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "-n",
        "--nms-threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def save_human(res, img, i: str):
    ins = res["instances"]
    clss = ins.pred_classes.cpu().numpy()
    if ins.has("pred_masks"):
        # print(ins[ins.gt_boxes])
        human_instances = ins[ins.pred_classes==0]
        for x in range(len(human_instances)):
            print(img[:-4])
            score = human_instances.scores.cpu().numpy()[x]
            background  = np.zeros_like(img)
            print(background.shape)
            W , H = background.shape[0], background.shape[1]
            print(f'height {H}, Width : {W}')
            bit_mask = human_instances[x].pred_masks
            background[bit_mask[0]]  = 255.0
            # cv2.imwrite(f"./crop_145/semantic/img{i}_instance_{x}.png", background)
            cv2.imwrite(f"./seg-dung/{img[:-4]}_instance_{x}_score_{score}.png", background)
            
            #Save bounding box
            bbox = masks_to_boxes(bit_mask).numpy()
            # print(bbox.shape)
            x_min, y_min, x_max, y_max = int(bbox[0,0]), int(bbox[0,1]), int(bbox[0,2]), int(bbox[0,3]), 
            print(f' xmin = {x_min}, xmax= {x_max}, ymin = {y_min}, ymax = {y_max}')
            # cv2.rectangle(background, (x_min, y_min), (x_max, y_max), color=(0, 0, 255))
            #crop = img[x_min : x_max,y_min : y_max]
            im_pil = Image.fromarray(img)
            try: 
                crop = im_pil.crop((x_min, y_min, x_max, y_max ))
                imgarr = np.array(crop)
                cv2.imwrite(f"./result-dung/2207_crop/{img[:-4]}_instance_{x}_score_{score}.png", imgarr)
                print(crop)
                
            except: 
                print(f"instance invalid")

            name_file = i + ".txt"
            #add instance to txt file
            # f = open("/home/huynhmink/Desktop/DUNG/yolov7_fake/yolov7/predict/" + name_file, 'a+')
            # # 
            # f.write("Person {} {} {} {} {}_ins{} \n".format(x_min, y_min, x_max, y_max, i, x))
            # f.close()

            # cv2.imwrite(f"./crop_145/semantic_with_boxes/img{i}_instance_{x}_score_{score}.png", background)
            # cv2.imwrite(f"./seg/img{i}_instance_{x}_score_{score}.png", background)


        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
                
        try:
            img = vis_bitmasks_with_classes(
                img, clss, bit_masks, force_colors=None, draw_contours=True, alpha=0.6, thickness=2
            )
        except ZeroDivisionError:
            img = img

        return img


def vis_res_fast(res, img, class_names, colors, thresh):
    ins = res["instances"]
    bboxes = None
    if ins.has("pred_boxes"):
        bboxes = ins.pred_boxes.tensor.cpu().numpy()
    scores = ins.scores.cpu().numpy()
    clss = ins.pred_classes.cpu().numpy()
    if ins.has("pred_bit_masks"):
        human_instances = ins[ins.pred_classes==0]
        for x in range(len(human_instances)):
            # print('pred_num',x)
            # print(human_instances[x].pred_masks.shape)
            background  = np.zeros_like(img)
            bit_mask = human_instances[x].pred_bit_masks
            background[bit_mask[0]]  = 255.0

        bit_masks = ins.pred_bit_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
            
        # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks, force_colors=colors, mask_border_color=(255, 255, 255), thickness=2)
        img = vis_bitmasks_with_classes(
            img, clss, bit_masks, force_colors=None, draw_contours=True, alpha=0.8
        )
    
    if ins.has("pred_masks"):
        human_instances = ins[ins.pred_classes==0]
        for x in range(len(human_instances)):
            # print('pred_num',x)
            # print(human_instances[x].pred_masks.shape)
            background  = np.zeros_like(img)
            bit_mask = human_instances[x].pred_masks
            background[bit_mask[0]]  = 255.0
            # cv2.imwrite(f"./result/semantic/instance_{x}.png", background)

        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
                
        try:
            img = vis_bitmasks_with_classes(
                img, clss, bit_masks, force_colors=None, draw_contours=True, alpha=0.6, thickness=2
            )
        except ZeroDivisionError:
            img = img

    thickness = 1 if ins.has("pred_bit_masks") else 2
    font_scale = 0.3 if ins.has("pred_bit_masks") else 0.4
    if bboxes:
        img = visualize_det_cv2_part(
            img,
            scores,
            clss,
            bboxes,
            class_names=class_names,
            force_color=colors,
            line_thickness=thickness,
            font_scale=font_scale,
            thresh=thresh,
        )
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = cfg.DATASETS.CLASS_NAMES
    predictor = DefaultPredictor(cfg)

    print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    colors = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in range(cfg.MODEL.YOLO.CLASSES)
    ]
    conf_thresh = cfg.MODEL.YOLO.CONF_THRESHOLD
    print("confidence thresh: ", conf_thresh)

    iter = ImageSourceIter(args.input)
    i = 0
    while True:
        i = i+1
        im = next(iter)
 
        if isinstance(im, str):
            im = cv2.imread(im)
        res = predictor(im)
        count = len(res["instances"])
        a = {}
        a = res['instances']
        # for x in range(count):
        #     print('pred_num',x)
        #     print (a[x].pred_masks)
        if count != 0:
          print(f'Number of masks: {len(res["instances"].pred_masks)}')
          # print('pred_mask :',res["instances"].pred_masks)
          # boxes = masks_to_boxes(res["instances"].pred_masks)
      
          
          # header = ['X_tl', 'Y_tl', 'X_br', 'Y_br']
          # with open('./bbox_info', 'w') as f:
          #     writer = csv.writer(f)
          #     writer.writerow(header)
          #     writer.writerow(boxes)

          # print(f'Number of masks: {len(res["instances"].pred_boxes)}')
          # print(f'Number of masks: {len(res["instances"].pred_bit_masks)}')
          # print(f'Number of instances: {len(res["instances"])}')
          # print(f'Number of masks: {len(res["instances"].pred_masks)}')
          # print(f'Shape of masks: {res["instances"].pred_masks.shape}')
          # res = vis_res_fast(res, im, class_names, colors, conf_thresh)
          index_image = '{:0>6}'.format(str(i))
          res = save_human(res, im, index_image)
          cv2.imwrite(f"./final/Final_result_{i}.jpg", res)
          # iter.video_writter.write(res)
          # cv2.imshow('frame', res)
          # cv2.imshow("frame", res)

        if iter.video_mode:
            cv2.waitKey(1)
        else:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

   