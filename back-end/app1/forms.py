from django import forms
from .models import Image


# 表单
class ImageForm(forms.ModelForm):
    """Form for the image model"""

    class Meta:
        model = Image
        fields = ('title', 'image')     # 关联到Image模型的表单对象，该表单与该表格绑定，需要填写的属性列为title和image
