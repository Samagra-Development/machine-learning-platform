from django.urls import path
from . import views

urlpatterns = [
    path('datasets',views.get_datasets,name='datasets'),
    path('datasets/<str:pk>',views.get_features,name='features'),
    path('models',views.get_all_model,name='model'),
    path('models/create',views.create_model,name='create_model'),
    path('models/<str:pk>',views.get_model,name='specific_model'),
]