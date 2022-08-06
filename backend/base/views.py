from rest_framework.decorators import api_view
from rest_framework.response import Response

from backend.settings import BASE_DIR
import os

# Create your views here.


@api_view(['GET'])
def get_datasets(request):
    available_dataset = os.listdir(f'{BASE_DIR}/utils/Data')
    return Response(data=available_dataset,status=200)