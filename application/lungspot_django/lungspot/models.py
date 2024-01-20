from django.db import models
from django.utils import timezone
import os

class Result(models.Model):
    input_img = models.ImageField(upload_to='input/')
    gradcam_img = models.ImageField(upload_to='output-gradcam/', null=True)
    prediction = models.CharField(max_length=10)
    created_in = models.DateTimeField(default=timezone.now)
    
    def delete(self, *args, **kwargs):
        if self.input_img:
            self.input_img.delete()
        if self.gradcam_img:
            self.gradcam_img.delete()
        super().delete(*args, **kwargs)