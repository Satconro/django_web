import os
import sys
from pathlib import Path
from django.db import models

# 建立3个数据表对象，分别存储原始图像、增强后的水下图像和包含目标检查结果的水下图像
class Image_o(models.Model):
    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='images/original/%Y/%m/%d/', blank=False)  # 使用ImageField需要 Pillow 库

    def __str__(self):
        return self.title


class Image_e(models.Model):
    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='images/enhanced/%Y/%m/%d/', blank=False)  # 使用ImageField需要 Pillow 库

    def __str__(self):
        return self.title


class Image_d(models.Model):
    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='images/detection/%Y/%m/%d/', blank=False)  # 使用ImageField需要 Pillow 库

    def __str__(self):
        return self.title


