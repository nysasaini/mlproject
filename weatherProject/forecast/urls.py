from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_view, name='weather'),

    path(
        'weather/location/',
        views.weather_by_location,
        name='weather_by_location'
    ),

    path(
        'api/weather/',
        views.get_weather_api,
        name='weather_api'
    ),
]