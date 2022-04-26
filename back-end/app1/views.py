import django
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt

from .forms import ImageForm


@csrf_exempt
def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance  # ImageModel实例
            # return render(request, 'app1/index.html', {'form': form, 'img_obj': img_obj})
            json_dict = {
                'image_url': img_obj.image.url,
                'draw_url': img_obj.image.url,
                'image_info': {
                    "bus-01": [
                        "195\u00d7264",
                        0.822
                    ],
                },
            }
            return JsonResponse(json_dict)
        return Http404
    elif request.method == 'GET':
        form = ImageForm()
        return render(request, 'app1/templates/app1/index.html', {'form': form})