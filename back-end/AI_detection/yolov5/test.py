import os
import cv2
from pathlib import Path

from detect.AIDetector_pytorch import Detector


def test_detection():
    img = cv2.imread("test/source/1.jpg")
    detector = Detector()
    img, img_info = detector.detect(img)
    return img, img_info


if __name__ == '__main__':
    test_detection()
