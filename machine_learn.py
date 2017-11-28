# Group Members:
# Graham, Trisha, Jonah
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import argparse
from keras.models import load_model
import data_parse
from matplotlib import pyplot as plt
from keras import optimizers
from keras.optimizers import TFOptimizer
import os


def format_data(data):
    """
    Formats feature-set in order for it to be fed into neural net.
    :param data: features to format.
    :return: features x, labels y
    """
    m = len(data)
    n = len(data[0]) - 1
    x = np.zeros((m, n))
    y = np.zeros((m, 1))
    for i in range(m):
        x[i] = data[i][1:]
        y[i] = data[i][0]
    return x, y


def run_nnet(x, y, gpu):
    """
    Run neural net for power predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :return: model
    """
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Create model.
    model = Sequential()
    dim1 = len(x)
    dim2 = len(x[0])
    # Add the layers.
    model.add(Dense(dim1, input_dim=dim2, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(120, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(60, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(80, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(100, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(150, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(180, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(30, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(20, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='random_uniform'))
    # Set the optimizer.
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    # Compile model.
    model.compile(loss='mse', optimizer=sgd, metrics=["mae"])
    # Fit the model.
    model.fit(x, y, epochs=20, batch_size=100, verbose=2, validation_split=0.2)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', "-m", dest='model', action='store', required=True, help="path to model being used")
    args = parser.parse_args()
    model = load_model(args.model)
    d = data_parse.read_data("data.csv")[50100:60100]
    x, y = format_data(d)
    print("Evaluating model...")
    evaluation = model.evaluate(x=x, y=y, verbose=1, batch_size=300)
    print("Loss(mse): "+str(evaluation[0])+"     Mean Absolute Error: " + str(evaluation[1]))

    # Plot the predictions
    predictions = model.predict(x)
    plt.plot(predictions, 'r', label="prediction")
    plt.plot(y, 'g', label='actual', linewidth=.5)
    leg = plt.legend()
    plt.show()
