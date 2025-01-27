#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess
from paddleocr import PaddleOCR
from fast_plate_ocr import ONNXPlateRecognizer
def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser

def vis(img, bboxes, scores, cls, cls_conf=0.35, text_scale=0.5, padding=5, thickeness=1):
    total_text = []
    for box_val, score, cls_ids in zip(bboxes, scores, cls):
            print("CLass IDs", cls_ids, box_val)
            if score >= cls_conf:
                x0, y0, x1, y1 = map(int, box_val)
                roi = img[y0:y1, x0:x1]
                

                x0 = int(box_val[0])
                y0 = int(box_val[1])
                x1 = int(box_val[2])
                y1 = int(box_val[3])

                text = ''
                result = ocr.ocr(roi, cls=False)# det=True,
                for idx in range(len(result)):
                    res = result[idx]
                    if res != None:
                        for line in res:
                            text += line[1][0] + " "
                if text == "":
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    output = fastocr.run(roi_gray)
                    text = output[0]

                if text != "":
                    total_text.append(text)
                    bg_color = (51, 159, 255)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    txt_size = cv2.getTextSize(text, font, text_scale, thickeness)[0]
                    cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, 2)

                    cv2.rectangle(
                        img,
                        (x0, y0 - txt_size[1] - padding - 5),
                        (x0 + txt_size[0] + 2, y0 - 5),
                        bg_color,
                        -1
                    )
                    cv2.putText(img, text, (x0, y0 - int(0.5 * txt_size[1])), font, text_scale, (0, 0, 0),
                                thickness=thickeness, lineType=cv2.LINE_AA)
    return img, total_text

if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    fastocr = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')
      
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img , total_text = vis(origin_img, final_boxes, final_scores, final_cls_inds)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
