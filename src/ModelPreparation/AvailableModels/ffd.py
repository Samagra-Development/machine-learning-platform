# import tensorflow as tf

# class FeedForwardNeuralNet(tf.keras.Sequential):
#     def __init__(self,neurons):
#         super(FeedForwardNeuralNet,self).__init__()
#         for layer in self._ffd(neurons):
#             self.add(layer)
    
#     @staticmethod
#     def _ffd(neurons):
#         layers = []
#         for neuron in neurons:
#             layers.append(tf.keras.layers.Dense(neuron))
#             layers.append(tf.keras.layers.ReLU())
#         return layers[:-1]

ffd = f"""
    # creating input
    inputs = [
        keras.layers.Input(shape=(_FEATURE_DICT[key]["len_value"],), name=key)
        for key in _FEATURE_KEYS
    ]
    d = keras.layers.concatenate(inputs)

    # adding all the hidden neurons
    for neuron in neurons:
        d = keras.layers.Dense(neuron, activation=_ACTIVATION)(d)
    
    # creating output
    num_output = 0
    for label in _FEATURE_LABEL:
        num_output += _FEATURE_DICT[label]["len_value"]
    outputs = keras.layers.Dense(num_output)(d)

    # generate and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_OPTIMIZER,loss=_LOSS,metrics=_METRICS)

    # log model
    model.summary(print_fn=logging.info)
    return model
"""