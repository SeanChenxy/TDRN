from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, pair_collate, seq_detection_collate, VOC_CLASSES, VID_CLASSES, VID_CLASSES_name, UW_CLASSES
from .coco import COCODetection, coco_detection_collate, COCO_CLASSES
from .config import *
import cv2
import numpy as np

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        # if multi_flag:
        #     return base_transform(image, dim, self.mean), boxes, labels
        # else:
        return base_transform(image, self.size, self.mean), boxes, labels
