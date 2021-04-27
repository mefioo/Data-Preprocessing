import numpy as np
import Helpers as hp
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow import keras
from math import sqrt, floor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class LSTMNN:
    def __init__(self, set_name, data, units, epochs, hidden_layers=1, lags=24, batch_size=120, lr=0.0009, momentum=0.9, optimizer='adam', loss='mae', show=1):
        self.set_name = set_name
        self.units = units
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.lags = lags
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.loss = loss
        self.data = data
        self.expected = []
        self.x_train, self.y_train, self.x_test, self.y_test, self.scaler_y = self.split_to_train_and_test()
        self.show = show
        self.model_name = self.set_model_name()
        self.save = 0

    def set_model_name(self):
        model_name = 'model' + self.set_name + '_u' + str(self.units) + '_e' + str(self.epochs) + '_lr' + str(self.lr) + '.h5'
        return model_name

    def create_input_data(self):
        data = hp.transformDataIntoSeries(self.data)
        x, y = [], []
        for step in range(self.lags, len(data)):
            startStep = step - self.lags
            tmp = []
            for i in range(self.lags):
                tmp.append(data[startStep + i])
                # tmp.append(step % 12)
                # tmp.append((floor(step % 12)/4))
            x.append(tmp)
            y.append(data[step])
        return np.array(x), np.array(y)

    def scale_data(self, x, y):
        x, y = x.astype(float), y.astype(float)
        rows = x.shape[0]
        columns = x.shape[1]
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scalerX = scaler.fit(x)
        scalerY = scaler.fit(y)
        x = scalerX.transform(x)
        y = scalerY.transform(y)
        x = np.array(x).reshape(rows, columns)
        y = np.array(y).reshape(-1)
        return x, y, scaler

    def scale_data_back(self, predictions):
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions)
        predictions = np.array(predictions).reshape(-1)
        predictions = [x if x > 0 else 0 for x in predictions]
        return predictions

    def split_to_train_and_test(self):
        x, y = self.create_input_data()
        years = 6
        split = 12 * years
        self.data = np.array(np.column_stack((x, y)))
        self.train = self.data[:-split]
        self.test = self.data[-split:]
        self.expected = y[-split:]
        x, y, scaler = self.scale_data(x, y)
        x_train, y_train = x[:-split], y[:-split]
        x_test, y_test = x[-split:], y[-split:]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
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
            model.add(LSTM(self.units, activation='relu', input_shape=(self.x_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(loss=self.loss, optimizer=self.choose_optimizer())
        es = keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, verbose=0, mode='auto',
        )
        history_callback = model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[es], verbose=0, shuffle=False)
        lossHistory = history_callback.history["loss"]
        return model, lossHistory

    def fit_and_compare(self):
        model, lossHistory = self.fit_model()
        if self.save:
            model.save(self.model_name)
        output = model.predict(self.x_test, batch_size=self.batch_size)
        predictions = hp.changeColumnIntoSeries(output.transpose()[0]).values
        predictions = hp.roundUpData(self.scale_data_back(predictions))
        # tmp = self.scale_data_back(self.y_test)
        expected = self.expected
        rmse = sqrt(mean_squared_error(expected, predictions))
        print(rmse)
        # print(self.expected)
        # print(tmp)
        return predictions, expected, lossHistory

    def show_plot(self, predictions, expected, loss_history):
        plt.plot(pd.Series(loss_history))
        if self.save:
            plt.savefig(self.model_name[:-3] + '_loss')
        plt.show()
        plt.plot(predictions, label=f'Predicted {self.units}, {self.epochs}')
        plt.plot(expected, label='Original')
        plt.legend(loc='best')
        plt.title(f'Prediction for {self.set_name} set')
        if self.save:
            plt.savefig(self.model_name[:-3] + '_prediction')
        plt.show()

    def single_LSTM(self):
        predictions, expected, lossHistory = self.fit_and_compare()
        self.show_plot(predictions, expected, lossHistory)
