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

@api_view(['POST'])
def create_model(request):
    data = request.data
    try:
        #assert data["model_type"] in available_models.keys()
        model = MlModel.objects.create(
            title = data['title'],
            dataset = data['dataset'],
            features = data['features'],
            label = data['label'],
            model_type = data["model_type"],
        )
        if "description" in data.keys():
            model.description = data['description']
            model.save()
            
        fin_data = MlModelSerializer(model,many=False).data
        return Response(fin_data,status=200)
    except:
        return Response({'message':'Please provide all the required columns : title,dataset,features,label'})
    
@api_view(['GET'])
def get_model(request,pk):
    try:
        models = MlModel.objects.get(id=pk)
        data = MlModelSerializer(models,many=False).data
        return Response(data,status=200)
    except:
        return Response({'message':"Model with given id doesn't exist"},status=400)
    
@api_view(['GET'])
def train_model(request,pk):
    try:
        model = MlModel.objects.get(id=pk)
    except:
        return Response({'message':"Model with given id doesn't exist"},status=400)
    
    id = model.id
    dataset = model.dataset
    features = model.features
    label = model.label
    model_type = model.model_type
    
    ROOT = f"{BASE_DIR}/Models/{str(id)}"
    PIPELINE_NAME = dataset.split('.')[0]
    # Output directory to store artifacts generated from the pipeline.
    PIPELINE_ROOT = os.path.join(ROOT,'pipelines', PIPELINE_NAME)
    # Path to a SQLite DB file to use as an MLMD storage.
    METADATA_PATH = os.path.join(ROOT,'metadata', PIPELINE_NAME, 'metadata.db')
    # Output directory where created models from the pipeline will be exported.
    SERVING_MODEL_DIR = os.path.join(ROOT,'serving_model', str(id))

    DATA_ROOT = f"{ROOT}/data"
    SCHEMA_PATH = f"{BASE_DIR}/Datasets/{PIPELINE_NAME}/schema"
    
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    os.mkdir(ROOT)
    os.mkdir(DATA_ROOT)
    
    dataset = f'{BASE_DIR}/utils/Data/{dataset}'
    shutil.copyfile(dataset,f"{DATA_ROOT}/data.csv")
    
    _module_file = f'{ROOT}/utils.py'
    schema_path = f"{SCHEMA_PATH}/schema.pbtxt"
    
    model_trainer = generate_model_trainer(
        features=features,
        labels=label,
        path_to_schema=schema_path,
        model_type = model_type
    )

    with open(_module_file,'w') as f:
        f.write(model_trainer)
        
    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            schema_path=SCHEMA_PATH,
            module_file=_module_file,
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_path=METADATA_PATH
        )
        
    )
    return Response({'message':'Model trained '},status=200)