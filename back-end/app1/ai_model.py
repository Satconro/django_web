import os
import sys
from pathlib import Path

import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))  # add ROOT to PATH

from AI_detection.enhance.pipeline import InferencePipe as Pipe1
from AI_detection.yolov5.pipeline import InferencePipe as Pipe2


class AIModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls._instance = object.__new__(cls)
            cls._instance.device = 'cuda'
            cls._instance.weight_en = os.path.join(BASE_DIR, 'AI_detection/weights/encoder.pt')
            cls._instance.weight_de = os.path.join(BASE_DIR, 'AI_detection/weights/decoder.pt')
            cls._instance.weight_yolo = os.path.join(BASE_DIR, 'AI_detection/weights/yolo.pt')
            cls._instance.pipe1 = Pipe1(cls._instance.device, cls._instance.weight_en, cls._instance.weight_de)
            cls._instance.pipe2 = Pipe2(cls._instance.device, cls._instance.weight_yolo)
        return cls._instance

    def __init__(self):
        super(AIModel, self).__init__()

    def enhance(self, img_path: str):
        # 返回Numpy格式图像
        enhanced_img = self.pipe1(img_path)
        return enhanced_img

    def detect(self, img_path: str):
        # 返回Numpy格式图像
        detected_img = self.pipe2(img_path)
        return detected_img
