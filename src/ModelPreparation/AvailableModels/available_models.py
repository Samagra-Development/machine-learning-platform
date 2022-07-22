from .ffd import ffd

available_models = {
    "classification" : {
        "model" : ffd,
        "loss" : "tf.keras.losses.CategoricalCrossentropy(from_logits=True)",
        "neurons" : [8,8],
        "optimizer" : "tf.keras.optimizers.Adam(1e-2)",
        "activation" : 'tf.keras.activations.relu',
        "metrics" : ['accuracy']
    },
    "regression" : {
        "model" : ffd,
        "loss" : "tf.keras.losses.MeanSquaredError()",
        "neurons" : [8,8],
        "optimizer" : "tf.keras.optimizers.Adam(1e-2)",
        "activation" : 'tf.keras.activations.relu',
        "metrics" : ["mean_absolute_error"]
    },
}