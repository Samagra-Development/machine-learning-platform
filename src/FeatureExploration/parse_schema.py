from tensorflow_data_validation.utils.schema_util import load_schema_text
from google.protobuf import json_format

def parse_schema_to_json(path_to_schema):
    """Parses the schema into readable json"""

    # loading and putting schema 
    schema = load_schema_text(path_to_schema)

    # Converting schema to a dictionary
    schema_message = json_format.MessageToDict(schema)

    # storing the features
    features = {}
    
    # add all the features
    for feature in schema_message['feature']:
        features[feature['name']] = {
            "type" : "categorical" if feature['type'] == 'BYTES' else feature['type'].lower(),
            "len_value" : 1
        }

    # add distinct values for categorical features
    for feature in schema_message['stringDomain']:
        features[feature['name']]["value"] = feature['value']
        features[feature['name']]["len_value"] = len(feature['value'])
    
    return features