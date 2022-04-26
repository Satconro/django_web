import os
from pathlib import Path

import os
import torch
from pipeline import UnderwaterImageEnhancementPipeline
from torchvision.utils import save_image

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


def test_enhancement():
    pipeline = UnderwaterImageEnhancementPipeline(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_folder=os.path.join(BASE_DIR, 'AI_detection/enhance/weights')
    )
    img = 'test/source/27.jpg'
    enhanced_img = pipeline(img_path=os.path.join(BASE_DIR, img))
    save_image(enhanced_img, os.path.join(BASE_DIR, 'test/result/27.jpg'))


if __name__ == '__main__':
    test_enhancement()
