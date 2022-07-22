ffd = f"""
    # creating input
    inputs = [
        keras.layers.Input(shape=(_FEATURE_DICT[key]["len_value"],), name=key)
        for key in _FEATURE_KEYS
    ]
    d = keras.layers.concatenate(inputs)

    # adding all the hidden neurons
    for neuron in _NEURONS:
        d = keras.layers.Dense(neuron, activation=_ACTIVATION)(d)
    
    # creating output
    num_output = _FEATURE_DICT[_FEATURE_LABEL]["len_value"]
    outputs = keras.layers.Dense(num_output)(d)

    # generate and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_OPTIMIZER,loss=_LOSS,metrics=_METRICS)

    # log model
    model.summary(print_fn=logging.info)
    return model
"""