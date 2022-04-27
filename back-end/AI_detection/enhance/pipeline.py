# 路径及sys.path处理
import os
import sys
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from . import utils
from .model import Enhancement_Encoder, Enhancement_Decoder


class SingleImageDataset(Dataset):
    def __init__(self, img_path):
        super(SingleImageDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.imgs = [img_path]
        self.len = len(self.imgs)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        img_name = self.imgs[index].split(os.sep)[-1]
        img_path = self.imgs[index]
        img = self.transform(Image.open(img_path))
        # 检查图片大小，限制在1920*1080以下
        img_size = img.shape
        if img_size[1] * img_size[2] > 1920 * 1080:
            trans = transforms.Resize([img_size[1] // 2, img_size[2] // 2])
            img = trans(img)
        return img_name, img


def get_dataloader_from_single_img(img_path):
    dataset = SingleImageDataset(img_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    return dataloader


class InferencePipe(nn.Module):
    def __init__(self, device, en_path, de_path):
        super(InferencePipe, self).__init__()
        # Device
        self.device = device
        # Model: load tested model and set mode to eval()
        self.encoder = Enhancement_Encoder().to(self.device).eval()
        self.decoder = Enhancement_Decoder().to(self.device).eval()
        # self.encoder = Enhancement_Encoder().half().to(self.device).eval()
        # self.decoder = Enhancement_Decoder().half().to(self.device).eval()
        # for layer in self.encoder.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.float()
        # for layer in self.decoder.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.float()
        utils.load_checkpoint(self.encoder, en_path, self.device)
        utils.load_checkpoint(self.decoder, de_path, self.device)
        # No grad calculation is needed in pipeline
        torch.set_grad_enabled(False)

    def forward(self, img_path):
        dataloader = get_dataloader_from_single_img(img_path)
        for name, img in dataloader:
            # img = img.to(self.device).half()
            img = img.to(self.device)
            generated_img = self.decoder(*self.encoder(img))
            # 转换数据格式为Numpy
            generated_img = generated_img.mul(255).byte()  # 取值范围
            generated_img = generated_img.cpu().numpy().squeeze(0).transpose((1, 2, 0))  # 改变数据大小
            torch.cuda.empty_cache()
            return generated_img

