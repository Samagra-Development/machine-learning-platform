There are some major places that can be used to contribute to the project
# Frontend
The main goal of Machine Learning Platform is to provide a user friendly way for a user with no prior knowledge of machine learning to come train a model and then deploy it for day to day uses.<br>
Currently the user needs to use the APIs to access the features but wouldn't it be great if there was a frontend to make user interaction easier. Developing a frontend is one of the most important contribution you can make to the project.<br>
To start contributing to the frontend part. Create a directory called <code>frontend/</code> in the base of the project and get started.<br>
The frontend must have the following feature : <br>
<ul>
    <li> UI to train model </li>
    <li> Dashboard </li>
<ul>
You can make use of the APIs provided in API.md to connect your frontend.

# Backend
The app has a simple backend written in django and mainly uses DRF to provide REST Api. The backend of the project is present in <code>backend/</code> folder. If you are familiar with Django and think that the code can be better or you can add any feature. Make sure to make a PR with the proposed feature.<br>
<ul>
    <li>
        One of the places you can contribute is taking user inpute to overwrite the default hyperparameters for the model. Ensure that this is an optional feature for advanced users.
    </li>
    <li>
        You can connect the MlModel database with the User database and add authentication.
    </li>
</ul>
# ML Pipelines
AI is a fairly new field and new advancements are coming day by day. The frameworks are improving and new models are coming into picture. This part of the project needs constant contributions to keep it up to date with the modern standards. You can contribute to this part in the <code>backend/utils</code> directory. The pipelines are written in TensorflowExtended.
Here are some places where you can contribute to
## Available Models
Currently the platform supports only two kind of models. One for regression and one for classification. Both of them uses a neural network with variable width and depth as the model. You can add more model to it by adding a model with all it's hyperparameters in available_models.py <code>/backend/utils/ModelPreparation/AvailableModels/available_models.py</code>. You can describe the model in a seperate file and then import it to the available_model. TFX expects the model to be described in a certain way. You can take motivation from ffd.py <code>/backend/utils/ModelPreparation/AvailableModels/ffd.py</code> to create your new model.
## Dataset Diversity
Currentl the model only supports tabular data and it would be great to extend it for Image and text dataset. You will need to make few changes to make this happen. You might need to update your model for this you can refer to contributing to Available Models section of this file. Next you will need to make some transformation for your data for that you will need to change or play around with the module_file_generator.py file <code>/backend/utils/ModelPreparation/Trainers/module_file_generator.py</code> which is used to generate the code for generating model and performing transformation.
