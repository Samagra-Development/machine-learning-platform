from ..AvailableModels.available_models import available_models
from ...FeatureExploration.parse_schema import parse_schema_to_json

def generate_model_trainer(
    features,
    labels,
    path_to_schema='schema.pbtxt',
    model_type="classification",
    train_batch_size=20,
    eval_batch_size=10,
    epoch = 10):

    model = available_models[model_type]
    
    # generating model specific hyperparameters0, 3
    model_hyperparameters = ""
    for k,v in model.items():
        if k=="model":
            continue
        model_hyperparameters += f'_{k.upper()} = {v} \n'

    model_trainer = f"""
from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
import json

# description of features present in the dataset
_FEATURE_DICT = {parse_schema_to_json(path_to_schema)}

# features and label as provided by users
_FEATURE_KEYS = {features}
_FEATURE_LABEL = "{labels}"

# model specific hyperparameters
{model_hyperparameters}

# batch sizes
_EPOCH = {epoch}
_TRAIN_BATCH_SIZE = {train_batch_size}
_EVAL_BATCH_SIZE = {eval_batch_size}

# TFX Transform will call this function.
def preprocessing_fn(inputs):
    # tf.transform's callback function for preprocessing inputs.

    # Args:
    # inputs: map from feature keys to raw not-yet-transformed features.

    # Returns:
    # Map from string feature key to transformed feature.
    
    outputs = {{}}

    # Uses features defined in _FEATURE_KEYS only.
    for key in _FEATURE_KEYS+[_FEATURE_LABEL]:
        # if the feature is categorical then do one hot encoding
        if _FEATURE_DICT[key]["type"] == "categorical":
            feature_keys = _FEATURE_DICT[key]["value"]
            initializer = tf.lookup.KeyValueTensorInitializer(
                                        keys=feature_keys,
                                        values=tf.cast(tf.range(len(feature_keys)), tf.int64),
                                        key_dtype=tf.string,
                                        value_dtype=tf.int64
                                    )
            feature_table = tf.lookup.StaticHashTable(initializer, default_value=-1)
            encoded = feature_table.lookup(inputs[key])
            depth = tf.cast(len(feature_keys),tf.int32)
            one_hot_encoded = tf.one_hot(encoded,depth)
            outputs[key] = tf.reshape(one_hot_encoded, [-1, depth])
        
        # else standorize it
        else:
            outputs[key] = tft.scale_to_z_score(inputs[key])

    return outputs

# this function will apply the same transform operation on training as well
# as serving request
def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if _FEATURE_LABEL in raw_features:
        transformed_label = transformed_features.pop(_FEATURE_LABEL)
        return transformed_features,transformed_label
    return transformed_features,None
    

def _get_serve_tf_examples_fn(model, tf_transform_output):
    # We must save the tft_layer to the model to ensure its assets are kept and tracked.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Expected input is a string which is serialized tf.Example format.
        feature_spec = tf_transform_output.raw_feature_spec()

        # Because input schema includes unnecessary fields like 'species' and
        # 'island', we filter feature_spec to include required keys only.
        required_feature_spec = {{
            k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
        }}
        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                                required_feature_spec)

        # Preprocess parsed input with transform operation defined in
        # preprocessing_fn().
        transformed_features, _ = _apply_preprocessing(parsed_features,
                                                        model.tft_layer)
        # Run inference with ML model.
        return model(transformed_features)

    return serve_tf_examples_fn

# Generates features and label for tuning/training.
def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    dataset = data_accessor.tf_dataset_factory(
                                    file_pattern,
                                    tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
                                    schema=tf_transform_output.raw_metadata.schema
                                )

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()

# build a model
def _build_keras_model() -> tf.keras.Model:
{model["model"]}

# TFX Trainer will call this function. to train the model
def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    history = model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs = _EPOCH)
        
    tmp_path = fn_args.transform_graph_path.split("pipelines")[0]
    metrics_path = f"{{tmp_path}}metrics.json"
    with open(metrics_path,"w") as f:
        json.dump(history.history,f)
    
    # NEW: Save a computation graph including transform layer.
    signatures = {{
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
    }}
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
"""
    return model_trainer