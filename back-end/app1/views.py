import os
import time
from io import BytesIO

import cv2
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse, Http404
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .ai_model import AIModel
from .forms import ImageForm
from .models import *


@csrf_exempt
def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance  # ImageModel实例
            # 0
            ai_model = AIModel()
            # 1
            en_img = ai_model.enhance(img_obj.image.path)
            en_img = trans_img_type(request.FILES['image'], en_img)
            img_obj1 = Image_e.objects.create(image=en_img)
            # 2
            dec_img, dec_info = ai_model.detect(img_obj1.image.path)
            dec_img = trans_img_type(request.FILES['image'], dec_img)
            img_obj2 = Image_d.objects.create(image=dec_img)
            # return
            json_dict = {
                'image_url': img_obj.image.url,
                'draw_url': img_obj2.image.url,
                'image_info': dec_info,
            }
            return JsonResponse(json_dict)
        return Http404
    elif request.method == 'GET':
        form = ImageForm()
        return render(request, 'app1/index.html', {'form': form})


def trans_img_type(ori, pic):
    # cv2转PIL
    pic = Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    # 先将PIL保存到IO流
    pic_io = BytesIO()
    pic.save(pic_io, 'jpeg')
    # 再转化为InMemoryUploadedFile数据
    pic_file = InMemoryUploadedFile(
        file=pic_io,
        field_name=None,
        name=ori.name,
        content_type=ori.content_type,
        size=ori.size,
        charset=None
    )
    return pic_file
