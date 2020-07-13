from django.urls import path
from . import views

urlpatterns = [
    path('', views.work().index,name='index'),
]

