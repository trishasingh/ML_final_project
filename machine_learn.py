from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import data_parse
from matplotlib import pyplot as plt

def run_nnet(d):
    """
    run nueral net for power predictions
    :param d: data to train on
    :return: model
    """
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
    model.add(Dense(50, kernel_initializer='uniform'))
    model.add(Dense(8, kernel_initializer='uniform'))
    model.add(Dense(100, kernel_initializer='uniform'))
    model.add(Dense(30, kernel_initializer='uniform'))
    model.add(Dense(20, kernel_initializer='uniform'))
    model.add(Dense(1, kernel_initializer='uniform'))
    # Compile model
    model.compile(loss='mse', optimizer='rmsprop', metrics=["mae"])
    # Fit the model
    model.fit(x, y, epochs=15, batch_size=1000, verbose=2, validation_split=0.2)
    return model


if __name__ == "__main__":
    model = load_model("models/model_2017-11-27_13_06_18.h5")
    d = data_parse.read_data("data.csv")[5100:]
    m = len(d)
    n = len(d[0]) - 1
    x = np.zeros((m, n))
    y = np.zeros((m, 1))
    for i in range(m):
        x[i] = d[i][1:]
        y[i] = d[i][0]
    print("Evaluating model...")
    #evaluation = model.evaluate(x=x, y=y, verbose=1, batch_size=300)
    #print("Loss(mse): "+str(evaluation[0])+"     Mean Absolute Error: " + str(evaluation[1]))
    predictions = model.predict(x)
    plt.plot(predictions,'r', y, 'g')
    plt.show()
