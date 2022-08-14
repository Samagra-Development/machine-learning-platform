# API
The APIs that are exposed for the platform are : 

## GET /api/datasets 
Returns all available datasets.
Looks in the dataset folder to see al the available folder and then returns it in form of a list.

## GET /api/datasets/{str:id} 
Returns feature related to a specific dataset.
Runs the SchemaGenerator to create schema.pbtxt and then reads to return all the available feature in form of dictionary.

## GET /api/models 
Returns all the available models.
Reads the MlModel database to achieve this.

## POST /api/models/create 
Create a new model

## GET /api/models/available 
Different type of model available for training.
Reads the available_model.py file to achieve this.

## GET /api/models/{str:id} 
Information about a specific model

## GET /api/models/{str:id}/train 
Train a specific model
Runs the training pipeline to achieve this.

## POST /api/models/{str:id}/predict 
Use a trained model to make prediction
