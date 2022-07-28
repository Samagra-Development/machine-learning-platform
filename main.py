import os
from tfx import v1 as tfx
from absl import logging
logging.set_verbosity(logging.INFO) 
import shutil
from src.FeatureExploration.schema_generator import SchemaGenerator
from src.FeatureExploration.parse_schema import parse_schema_to_json
from src.ModelPreparation.Trainers.module_file_generator import generate_model_trainer
from src.ModelPreparation.Trainers.training_pipeline import _create_pipeline
from src.ModelPreparation.AvailableModels.available_models import available_models

def choose_dataset():
    files = os.listdir('./src/Data')
    for f in files:
        print(f"{f}\t")
    while True:
        selected_file = input("Enter the dataset you want to work on : ")
        if selected_file in files:
            return f"./src/Data/{selected_file}"
        print("Enter a valid dataset")
        
def choose_feature_and_label(path_to_schema):
    feature_info = parse_schema_to_json(path_to_schema)
    features = []
    label = None
    print("Press 0 to pass a column 1 to use the column as feature and 2 to choose the column as label")
    print("\t Column Name\t |\t info \t|\t Option \n")
    print("=========================================================\n")
    for k in feature_info.keys():
        types = feature_info[k]["type"]
        choice = int(input(f"\t{k}\t|\t{types}\t|\t").strip())
        print("=========================================================\n")
        if choice==1:
            features.append(k)
        elif choice==2:
            label = k
    return features,label

def get_model_type():
    print("Choose a type of model to train on. Available models are : ")
    for k in available_models.keys():
        print(k)
    while True:
        model = input("Choose a model : ")
        if model in available_models.keys():
            return model
        print("Please choose a valid model")

ROOT = "./demo"
PIPELINE_NAME = "penguin"
# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join(ROOT,'pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join(ROOT,'metadata', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join(ROOT,'serving_model', PIPELINE_NAME)

DATA_ROOT = f"{ROOT}/data"
SCHEMA_PATH = f"{ROOT}/schema"

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

os.mkdir(ROOT)
os.mkdir(DATA_ROOT)
os.mkdir(SCHEMA_PATH)
        
dataset = choose_dataset()
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
features,label = choose_feature_and_label(schema_path)

_module_file = f'{ROOT}/utils.py'

model_type = get_model_type()

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