
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home , name='mac-home'),
    path('about/', views.about , name='mac-about'),
]