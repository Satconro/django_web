from django.db import models


class Image(models.Model):
    title = models.CharField(max_length=200, blank=True)
    image = models.ImageField(upload_to='images/%Y/%m/%d/', blank=False)  # 使用ImageField需要 Pillow 库

    def __str__(self):
        return self.title
