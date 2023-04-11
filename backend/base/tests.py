from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase
from .models import MlModel
from .views import get_datasets
from base.views import get_all_model
from base.serializers import MlModelSerializer
import json

class GetDatasetsTest(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Create test data - MlModel instances
        MlModel.objects.create(
            title="Test Model 1",
            dataset="data.csv",
            features="feature1-|-feature2",
            label="label1",
            model_type="classification",
            description="This is a test model 1."
        )
        MlModel.objects.create(
            title="Test Model 2",
            dataset="penguin_data.csv",
            features="feature1-|-feature2-|-feature3",
            label="label2",
            model_type="regression",
            description="This is a test model 2."
        )
        MlModel.objects.create(
            title="Test Model 3",
            dataset="training_df.csv",
            features="feature1-|-feature2-|-feature3-|-feature4",
            label="label3",
            model_type="regression",
            description="This is a test model 3."
        )

    def test_get_datasets(self):
        # Define the expected dataset list
        expected_datasets = ['data.csv', 'penguin_data.csv', 'training_df.csv']

        # Send a GET request to the endpoint
        response = self.client.get(reverse(get_datasets))

        # Assert that the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert that the response data matches the expected dataset list
        self.assertEqual(response.data, expected_datasets)

class GetFeaturesTest(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Create test data (MlModel instances) if necessary
        self.mlmodel = \
                    MlModel.objects.create(
            title="Test Model 3",
            dataset="data.csv",
            features="feature1-|-feature2-|-feature3-|-feature4",
            label="label3",
            model_type="regression",
            description="This is a test model 3."
        )

    def test_get_features(self):
        # Define the expected features list
        expected_features = ['Year','State','Type','Length','Expense','Value']
        
        # Remove the '.csv' extension from the dataset field
        dataset_name = self.mlmodel.dataset.split('.')[0]

        # Send a GET request to the endpoint with the model id
        response = self.client.get(reverse('features', kwargs={'pk': dataset_name}))

        # Assert that the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert that the response data matches the expected features list
        self.assertEqual(sorted(response.data), sorted(expected_features))

class GetAllModelsTest(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Create test data (MlModel instances)
        self.mlmodel1 = \
                    MlModel.objects.create(
                title="Test Model 4",
                dataset="data.csv",
                features=['Length', 'Type', 'Value', 'State', 'Expense'],
                label="Length",
                model_type="regression",
                description="This is a test model 4."
            )

    def test_get_all_model(self):
        # Send a GET request to the 'get_all_model' endpoint
        response = self.client.get(reverse('model'))

        # Deserialize the response data
        response_data = MlModelSerializer([self.mlmodel1], many=True).data

        # Check the status code and content of the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, response_data)

class CreateModelTests(APITestCase):

    def test_create_model(self):
        url = reverse('create_model')
        data = {
            'title': 'Test Model',
            'dataset': 'data.csv',
            'features': ['Length', 'Type', 'Value', 'State', 'Expense'],
            'label': 'Length',
            'model_type': 'regression',
            'description': 'This is a test model'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(MlModel.objects.count(), 1)
        self.assertEqual(MlModel.objects.get().title, 'Test Model')

    '''
    @PRESENTWORK:
    1. The Following 2 tests require logic implementation in the views.py file.

    @TODO:
    1. Implement logic in views.py to handle the following 2 tests.
    2. Uncomment the following 2 tests and run the tests.
    '''
    # def test_create_model_missing_data(self):
    #     url = reverse('create_model')
    #     data = {
    #         'title': 'Test Model',
    #         'dataset': 'test_dataset.csv',
    #         'features': ['feature1', 'feature2'],
    #         'label': 'label',
    #     }
    #     response = self.client.post(url, data, format='json')
    #     self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    #     self.assertEqual(MlModel.objects.count(), 0)

    # def test_create_model_invalid_model_type(self):
    #     url = reverse('create_model')
    #     data = {
    #         'title': 'Test Model',
    #         'dataset': 'test_dataset.csv',
    #         'features': ['feature1', 'feature2'],
    #         'label': 'label',
    #         'model_type': 'invalid_type',
    #         'description': 'This is a test model'
    #     }
    #     response = self.client.post(url, data, format='json')
    #     self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    #     self.assertEqual(MlModel.objects.count(), 0)

class GetModelTestCase(APITestCase):

    def setUp(self):
        self.model = MlModel.objects.create(
            title="Test Model",
            dataset="test_dataset.csv",
            features="[feature1, feature2]",
            label="label",
            model_type="classification"
        )

    def test_get_model(self):
        url = reverse('specific_model', kwargs={'pk': self.model.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, MlModelSerializer(self.model).data)

    def test_get_model_not_found(self):
        non_existent_id = 'non-existent-id'
        url = reverse('specific_model', kwargs={'pk': non_existent_id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {'message': "Model with given id doesn't exist"})

class TrainModelTestCase(APITestCase):

    def setUp(self):
        self.model = \
                            MlModel.objects.create(
                title="Test Model",
                dataset="data.csv",
                features=['Length', 'Type', 'State', 'Expense'],
                label="Value",
                model_type="regression",
                description="This is a test model"
            )
        data = {
            'title': 'Test Model',
            'dataset': 'data.csv',
            'features': ['Length', 'Type', 'State', 'Expense'],
            'label': 'Value',
            'model_type': 'regression',
            'description': 'This is a test model'
        }

        # Remove the '.csv' extension from the dataset field
        dataset_name = self.model.dataset.split('.')[0]

        # Send a GET request to the endpoint with the model id
        response = self.client.get(reverse('features', kwargs={'pk': dataset_name}))

        # Assert that the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        response = self.client.post(reverse('create_model'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_train_model(self):
        url = reverse('train_model', kwargs={'pk': self.model.id})
        response = self.client.get(url)

        self.model.refresh_from_db()
        self.assertEqual(response.status_code, 200)

    def test_train_model_not_found(self):
        non_existent_id = 'non-existent-id'
        url = reverse('train_model', kwargs={'pk': non_existent_id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {'message': "Model with given id doesn't exist"})

class PredictTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()

        self.model = \
                            MlModel.objects.create(
                title="Test Model",
                dataset="data.csv",
                features=['Year','Length', 'Type', 'State', 'Expense'],
                label="Value",
                model_type="regression",
                description="This is a test model"
            )
        data = {
            'title': 'Test Model',
            'dataset': 'data.csv',
            'features': ['Year','Length', 'Type', 'State', 'Expense'],
            'label': 'Value',
            'model_type': 'regression',
            'description': 'This is a test model'
        }

        # Remove the '.csv' extension from the dataset field
        dataset_name = self.model.dataset.split('.')[0]

        # Send a GET request to the endpoint with the model id
        response = self.client.get(reverse('features', kwargs={'pk': dataset_name}))

        # Assert that the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        response = self.client.post(reverse('create_model'), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Train the test model
        url = reverse('train_model', kwargs={'pk': self.model.id})
        response = self.client.get(url)

        # self.model.refresh_from_db()
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        # Prepare input data for the predict function
        input_data = {
            "Length": [[0], [1]],
            "Type": [[1], [0], [0]],
            "State": [[0], [0], [0],[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            "Expense": [[0], [1]],
            'Year': [[1.20007078]]
        }

        # Send a POST request to the predict endpoint with the input data
        url = reverse('predict_model', kwargs={'pk': self.model.id})
        response = self.client.post(url, data=json.dumps(input_data), content_type='application/json')
        # Check the response status code and the contents of the response
        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.content)
        self.assertIn('Value', response_data)

