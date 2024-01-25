from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
from django.conf import settings
import numpy as np
import cv2
from django.core.files.base import ContentFile
from .utils import getGradCamImg
from .models import Result
import time
# Create your views here.

MODELNAME = 'lungspot1'

def get_modelname(request):
    time.sleep(1)
    return JsonResponse(data={'modelname': MODELNAME})

@csrf_exempt
def predict_case(request):
    categories = ['benign', 'malignant', 'normal']
    model_dir = settings.BASE_DIR / ('keras_models/' + MODELNAME + '.h5')
    model = load_model(str(model_dir))
    uploaded_file = request.FILES.get('file')
    img_arr = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
    # Resize the image
    img_size = 256
    img = cv2.resize(img, (img_size, img_size))
    img_a = np.array([img]).reshape(-1, img_size, img_size, 1)
    pred = model.predict(img_a, verbose=False)
    pred_bool = np.argmax(pred, axis=1)
    output_case = categories[pred_bool[0]]
    prev_results = Result.objects.all()
    for res in prev_results:
        res.delete()
    result = Result(input_img=uploaded_file, prediction=output_case)
    result.save()
    model.layers[-1].activation = None
    gradcam_img = getGradCamImg(result.input_img.path, img_a, model, 'conv2d_5')
    gradimg_name = "gradcam-" + str(int(time.time())) + '.png'
    result.gradcam_img = ContentFile(gradcam_img, gradimg_name)
    result.save()
    return JsonResponse(data={
        'case': output_case, 
        'input_image_url': result.input_img.url, 
        'gradcam_image_url': result.gradcam_img.url, 
    })


