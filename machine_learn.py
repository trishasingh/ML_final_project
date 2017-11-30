# Group Members:
# Graham, Trisha, Jonah
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import argparse
from keras.models import load_model
import data_parse
from matplotlib import pyplot as plt
from keras import optimizers
import os
from keras import backend as K
import time, datetime
from datetime import datetime, timedelta


def format_data(data):
    """
    Formats feature-set in order for it to be fed into neural net.
    :param data: features to format.
    :return: features x, labels y
    """
    m = len(data)
    n = len(data[0]) - 2
    x = np.zeros((m, n))
    y = np.zeros((m, 1))

    for i in range(m):
        x[i] = data[i][2:]
        y[i] = data[i][1]

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y


def run_nnet(x, y, gpu):
    """
    Run neural net for power predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :return: model
    """
    # Create model.
    model = Sequential()
    dim1 = len(x)
    dim2 = len(x[0])
    # Add the layers.
    # model.add(Dense(dim1, input_dim=dim2, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(120, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(60, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(80, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(100, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(150, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(300, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(180, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(30, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(20, kernel_initializer='random_uniform', activation='relu'))
    # model.add(Dense(1, kernel_initializer='random_uniform'))
    # # Set the optimizer.
    # #sgd = optimizers.SGD(lr=0.01, clipnorm=2.)#, momentum=0.1, nesterov=True)
    # sgd = optimizers.Adagrad(clipnorm=2.)
    # #sgd = optimizers.Adadelta(clipnorm=2.)
    # # Compile model.
    # model.compile(loss='mae', optimizer=sgd)#, metrics=["mae"])
    # if gpu:
    #     # Fit the model.
    #     # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
    #     model.fit(x, y, epochs=100, batch_size=4096, verbose=2)  # , validation_split=0.2)
    # else:
    #     # Fit the model.
    #     # Feel free to change this batch size.
    #     model.fit(x, y, epochs=20, batch_size=1000, verbose=2)  # , validation_split=0.2)

    # Tuning
    model.add(Dense(dim1, input_dim=dim2, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='random_uniform'))
    # Set the optimizer.
    #sgd = optimizers.SGD(lr=0.01, clipnorm=2.)#, momentum=0.1, nesterov=True)
    sgd = optimizers.Adagrad(clipnorm=2.)
    #sgd = optimizers.Adadelta(clipnorm=2.)
    # Compile model.
    model.compile(loss='mae', optimizer=sgd)#, metrics=["mae"])
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        model.fit(x, y, epochs=100, batch_size=4096, verbose=2)  # , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        model.fit(x, y, epochs=25, batch_size=1000, verbose =2, validation_split=0.2)
    return model


def add_generate_NN_features(x, data, holidays): # based off features used in Gajowniczek paper
    """
    Generate features for the data-set.
    :param data: parsed raw data
    :param holidays: parsed holiday info
    :return: features
    """
    if len(x) < 96*4:
        raise IndexError("Too Few x's")
    hour = data[0].hour
    # Booleans for hour of the day.
    for h in range(24):
        data.append(hour == h)
    # Booleans for day of week.
    wd = data[0].weekday()
    for k in range(7):
        data.append(wd == k)
    # Booleans for day of the month.
    md = data[0].day
    for j in range(31):
        data.append(md == j)
    # Booleans for month of the year.
    month = data[0].month
    for l in range(12):
        data.append(month == l)
    data.append(data[0].date() in holidays)
    # Past 24 hours of demand.
    d1 = []
    # Energy usage for each of the last 96 periods.
    # If it is one of the first 96 periods, fill in zeros.
    for p1 in range(96):
        d1.append(0)
    for pa in range(96):
        d1[pa] += float(x[-pa-1])
    for p2 in d1:
        data.append(p2)
    # Minimum load of last 12, 24, 48, 96 periods (3,6,12,24 hours).
    for pb in [12, 24, 48, 96]:
        d2 = [data_parse.MAX_LOAD]
        for pb1 in range(pb):

            d2.append(float(x[-pb1 - 1]))
        data.append(min(d2))
    # Maximum load of last 12, 24, 48, 96 periods (3,6,12,24 hours).
    for pb in [12, 24, 48, 96]:
        d2 = [0]
        for pb1 in range(pb):
            d2.append(float(x[-pb1 - 1]))
        data.append(max(d2))
    # Load of the same hour in all days of the previous week.
    pc = []
    for pc1 in range(6):
        pc.append(0)
        pc[pc1] = float(x[ - 96 * (pc1 + 1)])
    for pc2 in pc:
        data.append(pc2)
    # Load of the same hour on the same weekday in previous 4 weeks.
    pd = []
    for pd1 in range(4):
        pd.append(0)
        pd[pd1] = float(x[- 96 * 7 * (pd1 + 1)])
    for pd2 in pd:
        data.append(pd2)

    return data[1:]


def forward_predict(x, y, initial_date, model, periods):
    """
    Propagate predictions forward to forecast demand.
    :param x: features
    :param y: labels
    :param model: model to use
    :param periods: number of examples forward to forecast
    :return:
    """
    predictions = []
    holidays = set(data_parse.parse_holidays("USBankholidays.txt"))
    for i in range(periods):
        p = model.predict(x)
        last = p[-1]
        new = [initial_date+timedelta(minutes=15)]
        initial_date = initial_date+timedelta(minutes=15)
        # Add each prediction
        predictions.append(last)
        if i > 0:
            y = np.append(y, [last])
            y = y.astype(np.float32)
        new = add_generate_NN_features(y, new, holidays)
        x = np.append(x, [new], axis=0)
        x = x.astype(np.float32)
        print("Forecast number: " + str(i+1)+" of "+str(periods)+" Predicted val: "+str(last))
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', "-m", dest='model', action='store', required=True, help="path to model being used")
    args = parser.parse_args()
    model = load_model(args.model)
    start = 50000
    stop = 70000
    d = data_parse.read_data("data.csv")[start:stop]
    x, y = format_data(d)
    print("Evaluating model...")
    evaluation = model.evaluate(x=x, y=y, verbose=1, batch_size=300)
    print("Loss(mae): "+str(evaluation))

    # Plot the predictions.
    periods = 96
    predictions = model.predict(x)
    forecast = forward_predict(np.copy(x[:(stop-start)//2]), np.copy(y[:(stop-start)//2]), d[(stop-start)//2][0], model, periods)
    plt.plot(predictions, 'r', label="prediction")

    plt.plot([((stop-start)//2) + i for i in range(len(forecast))], forecast, 'b', label="forecast")
    plt.plot(y, 'g', label='actual', linewidth=.5)
    leg = plt.legend()
    plt.show()
