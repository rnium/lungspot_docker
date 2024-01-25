from django.contrib import admin
from django.urls import path
from lungspot import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/modelname/', views.get_modelname, name="get_modelname"),
    path('api/predictcase/', views.predict_case, name="predict_case"),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)