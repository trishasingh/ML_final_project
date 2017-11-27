from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras import backend as K

def run_nnet(d, gpu, custom):
    """
    run nueral net for power predictions
    :param d: data to train on
    :param custom: use custom settings?
    :param gpu: use gpu?
    :return: model
    """
    #setup custom session
    if custom:
        num_cores = 4
        if gpu:
            num_GPU = 1
            num_CPU = 1
        else:
            num_CPU = 1
            num_GPU = 0

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
        session = tf.Session(config=config)
        K.set_session(session)

    m = len(d)
    n = len(d[0]) - 1
    x = np.zeros((m, n))
    y = np.zeros((m, 1))
    for i in range(m):
        x[i] = d[i][1:]
        y[i] = d[i][0]

    # create model
    model = Sequential()
    dim1 = len(x)
    dim2 = len(x[0])
    # add the layers
    model.add(Dense(dim1, input_dim=dim2, kernel_initializer='uniform'))
    model.add(Dense(15, kernel_initializer='uniform'))
    model.add(Dense(8, kernel_initializer='uniform'))
    model.add(Dense(20, kernel_initializer='uniform'))
    model.add(Dense(1, kernel_initializer='uniform'))
    # Compile model
    model.compile(loss='mse', optimizer='rmsprop', metrics=["mae"])
    # Fit the model
    model.fit(x, y, epochs=10, batch_size=200, verbose=2, validation_split=0.2)
    # model.evaluate(x_cv, y_cv, batch_size=20)
    # calculate predictions
    #predictions = model.predict(x)
    # round predictions
    #rounded = [round(x[0]) for x in predictions]
    #print(rounded)
    return model