import os
import cv2
from pathlib import Path

import torch.nn as nn

from .detect.AIDetector_pytorch import Detector


class InferencePipe(nn.Module):
    def __init__(self, device, weights_path):
        super(InferencePipe, self).__init__()
        self.detector = Detector(device, weights_path)

    def forward(self, img_path):
        img = cv2.imread(img_path)
        img, img_info = self.detector.detect(img)
        return img, img_info
