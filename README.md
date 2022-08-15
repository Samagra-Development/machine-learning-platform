# Machine Learning Platform

This goal of this project is to build a seamless machine learning solution with a feature store to use/resuse/update features easily and use integration with training, deplyoying, prediction services to give predictions to the user over gov-tech usecases.

This project is build with **Hopsworks Opensource library**, **Tensorflow**, **Tensorflow Extended**, **Django** and **Django REST Framwork**.

# How to set up the ML platform?
- You can follow the following steps to setup the project locally:
<ol>
    <li> 
        Fork the Repo. 
    </li>
    <li> 
        Clone the repo locally using <code>git clone https://github.com/{your-username}/machine-learning-platform.git</code> 
    </li>
    <li> 
        Go into the directory using <code>cd machine-learning-platform</code> 
    </li>
    <li> 
        Create a new virtual enviornment <code>python3 -m venv env</code>. If you don't have virtual enviornment install. Install using <code>pip install virtualenv</code>
    </li>
    <li>
        Install the dependencies using <code>pip3 install -r requirments.txt</code>
    </li>
    <li>
        Ready your database for migration <code>python3 manage.py makemigrations</code>
    </li>
    <li>
        migrate <code>python3 manage.py migrate --run-syncdb</code>
    </li>
    <li>
        Start your backend development server <code>python3 manage.py runserver</code>
    </li>
</ol>

To set up the feature store with the cloud, follow the detailed documentation [here](https://docs.google.com/document/d/19OANtsMTX6f5xBhio9-m9tKORE-lg2OFT8-MpEXqva8/edit?usp=sharing).

# APIs associated with ML Platform
The APIs that are exposed for the platform are : 

### GET /api/datasets 
Returns all available datasets.
Looks in the dataset folder to see al the available folder and then returns it in form of a list.

### GET /api/datasets/{str:id} 
Returns feature related to a specific dataset.
Runs the SchemaGenerator to create schema.pbtxt and then reads to return all the available feature in form of dictionary.

### GET /api/models 
Returns all the available models.
Reads the MlModel database to achieve this.

### POST /api/models/create 
Create a new model

### GET /api/models/available 
Different type of model available for training.
Reads the available_model.py file to achieve this.

### GET /api/models/{str:id} 
Information about a specific model

### GET /api/models/{str:id}/train 
Train a specific model
Runs the training pipeline to achieve this.

### POST /api/models/{str:id}/predict 
Use a trained model to make prediction

# Deploying the Machine Learning Platform

Deploying the ML platform app is fairly easy; you can contarise the Django app and then deploy it. This kind of deployment won’t let you deal with the jobs management for model training as Django will automatically run it in one of the threads. To run the training on a different server you just need to change the LocalDagRunner in backend/base/views.py with the pipeline runner for your specific server. TFX comes with inbuilt runner for kubeflow based server which is KubeflowDagRunner. The feature store once connected with AWS/Azure/GCP provides an easy way to ingest features into the pipelines.

In case of any queries, please feel free to reach out to: [Janhavi Lande](https://www.linkedin.com/in/janhavi12/), [Pratyaksh Singh](https://www.linkedin.com/in/psn0w/).
