from django.urls import path
from . import views

urlpatterns = [
    path('datasets',views.get_datasets,name='datasets'),
    path('datasets/<str:pk>',views.get_features,name='features'),
]