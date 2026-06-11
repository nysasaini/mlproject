from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_view, name='weather'),
    path('get-weather/', views.get_weather_api),
]