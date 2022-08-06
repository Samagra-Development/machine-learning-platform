from rest_framework.decorators import api_view
from rest_framework.response import Response

from backend.settings import BASE_DIR
import os
import shutil
from tfx import v1 as tfx
import tensorflow as tf
import numpy as np

from utils.FeatureExploration.schema_generator import SchemaGenerator
from utils.FeatureExploration.parse_schema import parse_schema_to_json
from utils.ModelPreparation.AvailableModels.available_models import available_models
from utils.ModelPreparation.Trainers.module_file_generator import generate_model_trainer
from utils.ModelPreparation.Trainers.training_pipeline import _create_pipeline

from .models import MlModel
from .serializers import MlModelSerializer

@api_view(['GET'])
def get_datasets(request):
    available_dataset = os.listdir(f'{BASE_DIR}/utils/Data')
    return Response(data=available_dataset,status=200)

@api_view(['GET'])
def get_features(request,pk):
    file_name = f'{pk}.csv'
    dataset = f'{BASE_DIR}/utils/Data/{file_name}'
    
    if not os.path.exists(dataset):
        return Response({'message':'Choose a valid dataset'},status=404)
    
    file_name = file_name.split('.')[0]
    ROOT = f'{BASE_DIR}/Datasets/{file_name}'
    PIPELINE_NAME = file_name
    DATA_ROOT = f"{ROOT}/data"
    SCHEMA_PATH = f"{ROOT}/schema"
    # Path to a SQLite DB file to use as an MLMD storage.
    METADATA_PATH = os.path.join(ROOT,'metadata', PIPELINE_NAME, 'metadata.db')
    # Output directory to store artifacts generated from the pipeline.
    PIPELINE_ROOT = os.path.join(ROOT,'pipelines', PIPELINE_NAME)
    
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    os.mkdir(DATA_ROOT)
    os.mkdir(SCHEMA_PATH)
    shutil.copyfile(dataset,f"{DATA_ROOT}/data.csv")
    
    schema_generator = SchemaGenerator(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        metadata_path=METADATA_PATH
    )

    schema_generator.run()
    schema_generator.copy_to_directory(path_to_directory=SCHEMA_PATH)

    schema_path = f"{SCHEMA_PATH}/schema.pbtxt"
    
    feature_info = parse_schema_to_json(schema_path)
    return Response(data=feature_info,status=200)

@api_view(['GET'])
def get_all_model(request):
    models = MlModel.objects.all()
    data = MlModelSerializer(models,many=True).data
    return Response(data,status=200)