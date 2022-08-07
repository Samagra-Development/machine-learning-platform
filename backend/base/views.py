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
    with open(f'{ROOT}/metrics.json') as f:
        model_metrics = json.load(f)
    
    # update model
    model.train_loss = round(model_metrics["loss"][-1],4)
    model.train_accuracy = round(model_metrics["accuracy"][-1],4)
    model.test_loss = round(model_metrics["val_loss"][-1],4)
    model.test_accuracy = round(model_metrics["val_accuracy"][-1],4)
    model.save()
    
    model = MlModel.objects.get(id=pk)
    data = MlModelSerializer(model,many=False).data
    return Response(data,status=200)

@api_view(['GET'])
def get_model_type(request):
    return Response(available_models.keys(),status=200)

@api_view(['POST'])
def predict(request,pk):
    trained_models = os.listdir(f'{BASE_DIR}/Models')
    try:
        model = MlModel.objects.get(id=pk)
    except:
        return Response({'message':"Model with given id doesn't exist"},status=400)
    if pk not in trained_models:
        return Response({'message':'Model is not trained yet'},status=200)
    data = request.data
    dataset = model.dataset
    features = model.features
    features = [x.replace("'"," ").replace('"'," ").strip() for x in features.strip('][').split(',')]
    print(features)
    label = model.label
    latest = max([int(x) for x in os.listdir(f'{BASE_DIR}/Models/{pk}/serving_model/{pk}')])
    
    inp = []
    file_name = dataset.split('.')[0]
    schema_data = parse_schema_to_json(f'{BASE_DIR}/Datasets/{file_name}/schema/schema.pbtxt')
    
    for feature in features:
        if feature not in data.keys():
            return Response({'message':f'Must provide for feature : {feature}'},status=400)
        
        feat_value = data[feature]
        
        if not isinstance(feat_value,list):
            return Response({'message':'Input must be a list'},status=400)
        feature_len = schema_data[feature]["len_value"]
        if feature_len != len(feat_value):
            return Response({'message':f'input for {feature} should be {feature_len}'},response=400)
        inp.append(tf.convert_to_tensor(np.array(feat_value,ndmin=2,dtype=np.float32)))
        
    trained_model = tf.saved_model.load(f'{BASE_DIR}/Models/{pk}/serving_model/{pk}/{latest}')
    
    output = trained_model(inp).numpy()[0]
    
    label_value = schema_data[label]
    if label_value["len_value"] == 1:
        return Response({label:output},status=200)
    fin_output = {}
    for (a,b) in zip(label_value["value"],output):
        fin_output[a] = b
        
    return Response({label:fin_output},status=200)