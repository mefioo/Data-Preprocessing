import numpy as np
import Helpers as hp
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# input: t-1, t-2, ..., t-lags, prev_year_demand, t-1_prod1, t-1_prod2, t-1_prod3, t-1_prod4, t-1_prod5
def createInputData(series, prod_1, prod_2, prod_3, prod_4, prod_5, lags=1):
    x, y = [], []
    for step in range(12, len(series)): # minus one year so that we can take monthly demend from previous year for first data
        startStep = step - lags
        tmp = []
        for i in range(lags):
            tmp.append(series[startStep + i])
        tmp.append(series[step - 12])
        tmp.append(prod_1[step - 1])
        tmp.append(prod_2[step - 1])
        tmp.append(prod_3[step - 1])
        tmp.append(prod_4[step - 1])
        tmp.append(prod_5[step - 1])
        x.append(tmp)
        y.append(series[step])
    return np.array(x), np.array(y)


def scaleData(x, y):
    x, y = x.astype(float), y.astype(float)
    #x = np.interp(x, (x.min(), x.max()), (-1, 1))
    #y = np.interp(y, (y.min(), y.max()), (-1, 1))
    return x, y


def fitModel(x_train, y_train, batch_size, epochs, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    return model


def MLP(data, prod_1, prod_2, prod_3, prod_4, prod_5, epochs, showPlot=1):
    size = 10
    n = 3
    batch_size = 4
    data = hp.transformDataIntoSeries(data)
    prod_1 = hp.transformDataIntoSeries(prod_1)
    prod_2 = hp.transformDataIntoSeries(prod_2)
    prod_3 = hp.transformDataIntoSeries(prod_3)
    prod_4 = hp.transformDataIntoSeries(prod_4)
    prod_5 = hp.transformDataIntoSeries(prod_5)
    x, y = createInputData(data, prod_1, prod_2, prod_3, prod_4, prod_5, lags=n)
    x, y = scaleData(x, y)
    x_train, y_train = x[:-24], y[:-24]
    x_test, y_test = x[-24:], y[-24:]
    model = fitModel(x_train, y_train, batch_size=batch_size, epochs=epochs, neurons=4)
    model.save('modelMLP.h5')
    output = model.predict(x_test, batch_size=batch_size)
    predictions = hp.changeColumnIntoSeries(output.transpose()[0]).values
    expected = hp.changeColumnIntoSeries(y_test).values
    rmse = sqrt(mean_squared_error(expected, predictions))
    print(rmse)
    if showPlot:
        plt.plot(expected, label='Original')
        plt.plot(predictions, color='red', label='predicted')
        plt.legend(loc='best')
        plt.title(f'Prediction for sth set')
        plt.show()
    return rmse
