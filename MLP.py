import numpy as np
import Helpers as hp
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MLPNN:
    def __init__(self, data, epochs, neurons, hidden_layers=1, lags=24, batch_size=24, lr=0.0001, momentum=0.8, optimizer='sgd', loss='mse'):
        self.neurons = neurons
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.lags = lags
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.loss = loss
        self.data = data
        self.x_train, self.y_train, self.x_test, self.y_test, self.scaler_y = self.split_to_train_and_test()

    def create_input_data(self):
        data = hp.transformDataIntoSeries(self.data)
        x, y = [], []
        for step in range(24, len(data)):
            startStep = step - self.lags
            tmp = []
            for i in range(self.lags):
                tmp.append(data[startStep + i])
            # tmp.append(series[step - 12])
            x.append(tmp)
            y.append(data[step])
        return np.array(x), np.array(y)

    def scale_data(self, x, y):
        x, y = x.astype(float), y.astype(float)
        rows = len(x)
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scalerX = scaler.fit(x)
        scalerY = scaler.fit(y)
        x = scalerX.transform(x)
        y = scalerY.transform(y)
        x = np.array(x).reshape(rows, self.lags)
        y = np.array(y).reshape(-1)
        return x, y, scaler

    def scale_data_back(self, predictions):
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions)
        predictions = np.array(predictions).reshape(-1)
        return predictions

    def split_to_train_and_test(self):
        x, y = self.create_input_data()
        x, y, scaler = self.scale_data(x, y)
        x_train, y_train = x[:-24], y[:-24]
        x_test, y_test = x[-24:], y[-24:]
        return x_train, y_train, x_test, y_test, scaler

    def choose_optimizer(self):
        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum, nesterov=True)
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=self.lr)
        return opt

    def fit_model(self):
        model = Sequential()
        for i in range(self.hidden_layers):
            model.add(Dense(self.neurons, activation='sigmoid', input_dim=self.x_train.shape[1]))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=self.choose_optimizer())
        history_callback = model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, shuffle=False)
        lossHistory = history_callback.history["loss"]
        return model, lossHistory

    def fit_and_compare(self):
        model, lossHistory = self.fit_model()
        # model.save('modelMLP.h5')
        output = model.predict(self.x_test, batch_size=self.batch_size)
        predictions = hp.changeColumnIntoSeries(output.transpose()[0]).values
        predictions = self.scale_data_back(predictions)
        expected = hp.changeColumnIntoSeries(self.y_test).values
        expected = self.scale_data_back(expected)
        rmse = sqrt(mean_squared_error(expected, predictions))
        print(rmse)
        return predictions, expected, lossHistory

    def show_plot(self, predictions, expected, loss_history):
        plt.plot(pd.Series(loss_history))
        plt.show()
        plt.plot(predictions, label=f'predicted {self.neurons}, {self.epochs}')
        plt.plot(expected, label='Original')
        plt.legend(loc='best')
        plt.title(f'Prediction for sth set')
        plt.show()

    def single_MLP(self):
        predictions, expected, lossHistory = self.fit_and_compare()
        self.show_plot(predictions, expected, lossHistory)



# input: t-1, t-2, ..., t-lags, prev_year_demand
def createInputData(series, lags=1):
    x, y = [], []
    for step in range(24, len(series)): # minus one year so that we can take monthly demend from previous year for first data
        startStep = step - lags
        tmp = []
        for i in range(lags):
            tmp.append(series[startStep + i])
        #tmp.append(series[step - 12])
        x.append(tmp)
        y.append(series[step])
    return np.array(x), np.array(y)


def scaleData(x, y, lags):
    x, y = x.astype(float), y.astype(float)
    rows = len(x)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scalerX = scaler.fit(x)
    scalerY = scaler.fit(y)
    x = scalerX.transform(x)
    y = scalerY.transform(y)
    x = np.array(x).reshape(rows, lags)
    y = np.array(y).reshape(-1)
    return x, y, scalerY


def scaleBackData(x, scaler):
    x = np.array(x).reshape(-1, 1)
    x = scaler.inverse_transform(x)
    x = np.array(x).reshape(-1)
    return x


def fitModel(x_train, y_train, batch_size, epochs, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='sigmoid', input_dim=x_train.shape[1]))
    model.add(Dense(1))
    opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8, nesterov=True)
    model.compile(loss='mse', optimizer=opt)
    history_callback = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    lossHistory = history_callback.history["loss"]
    return model, lossHistory


def prepareData(data, n):
    data = hp.transformDataIntoSeries(data)
    x, y = createInputData(data, lags=n)
    x, y, scaler = scaleData(x, y, n)
    x_train, y_train = x[:-24], y[:-24]
    x_test, y_test = x[-24:], y[-24:]
    return x_train, y_train, x_test, y_test, scaler


def fitAndCompare(x_train, y_train, x_test, y_test, epochs, neurons, scaler):
    batch_size = 12
    model, lossHistory = fitModel(x_train, y_train, batch_size=batch_size, epochs=epochs, neurons=neurons)
    # model.save('modelMLP.h5')
    output = model.predict(x_test, batch_size=batch_size)
    predictions = hp.changeColumnIntoSeries(output.transpose()[0]).values
    predictions = scaleBackData(predictions, scaler)
    expected = hp.changeColumnIntoSeries(y_test).values
    expected = scaleBackData(expected, scaler)
    rmse = sqrt(mean_squared_error(expected, predictions))
    print(rmse)
    return predictions, expected, lossHistory


def MLP(data, epochsList, neuronsList, lags=24):
    x_train, y_train, x_test, y_test, scaler = prepareData(data, lags)
    for neurons in neuronsList:
        for epochs in epochsList:
            predictions, expected, lossHistory = fitAndCompare(x_train, y_train, x_test, y_test, epochs, neurons, scaler)
            if len(neuronsList) > 1:
                plt.plot(predictions, label=f'predicted {neurons}, {epochs}')
            else:
                plt.plot(pd.Series(lossHistory))
                plt.show()
                plt.plot(predictions, color='red', label='predicted')
        plt.plot(expected, label='Original')
        plt.legend(loc='best')
        plt.title(f'Prediction for sth set')
        plt.show()
    return lossHistory


def compareMLPs(data, parameters, lags=24):
    x_train, y_train, x_test, y_test, scaler = prepareData(data, lags)
    for neurons, epochs in parameters:
        predictions, expected, lossHistory = fitAndCompare(x_train, y_train, x_test, y_test, epochs, neurons, scaler)
        plt.plot(predictions, label=f'predicted {neurons}, {epochs}')
    plt.plot(expected, label='Original')
    plt.legend(loc='best')
    plt.title(f'Prediction for sth set')
    plt.show()
