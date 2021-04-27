import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
import numpy as np
import math


def preprocessDataForTwoColumnsAndNoZeros(data):
    result = []
    for row in data[1:]:
        tmp = []
        transaction = row[0].split(';')
        if (len(transaction) != 4):
            if row[-1].split(';')[-1] != '0':
                tmp.append(transaction[0])
                tmp.append(row[-1].split(';')[-2])
                result.append(tmp)
        else:
            if transaction[3] != '0':
                tmp.append(transaction[0])
                tmp.append(transaction[3])
                result.append(tmp)
    return result


def countAmountPerMonth(data):
    sum = 0
    month = '12'
    year = '2020'
    result = []
    for transaction in data:
        tmp = []
        date = transaction[0].split('.')
        if date[1] == month:
            sum += int(transaction[1])
        else:
            tmp.append(month + '.' + year)
            tmp.append(sum)
            result.append(tmp)
            sum = int(transaction[1])
            month = date[1]
            year = date[2]
    tmp = []
    tmp.append(month + '.' + year)
    tmp.append(sum)
    result.append(tmp)
    return result


def changeColumnIntoSeries(data):
    df = pd.Series(data)
    return df


def addMissingData(data):
    firstYear = int(data[0][0].split('.')[1])
    lastYear = int(data[-1][0].split('.')[1])
    finalData = []
    currentData = 0
    for year in range(lastYear - firstYear):
        for month in range(1, 13):
            if int(data[currentData][0].split('.')[0]) == month:
                finalData.append(data[currentData])
                currentData += 1
            else:
                tmp = []
                if month < 10:
                    tmp.append('0' + str(month) + '.' + str(firstYear + year))
                else:
                    tmp.append(str(month) + '.' + str(firstYear + year))
                tmp.append(int(0))
                finalData.append(tmp)
    return finalData

import random
def transformDataIntoSeries(data):
    chart = np.array(data)
    values = list(chart[:, 1].astype(int))
    # for i in range(10000):
    #     values.append(random.randint(70, 110))
    series = changeColumnIntoSeries(values)
    return series


def arimaModel(series, order, name, type):
    times = []
    X = series.values
    size = int(len(X) * 0.75)
    trainSet, testSet = X[:size], X[size:]
    print(f"Predicting set: {len(testSet)} samples.")
    historyValues = [x for x in trainSet]
    predictions = []
    for i in range(len(testSet)):
        start = time.time()
        model = ARIMA(historyValues, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        prediction = int(output[0])
        if prediction < 0:
            prediction = 0
        predictions.append(prediction)
        actualValue = testSet[i]
        if type == 'short':
            historyValues.append(actualValue)
        else:
            historyValues.append(prediction)
        end = time.time()
        times.append(end - start)
        print(f"{i + 1}/{len(testSet)} Predicted {prediction}, expected {actualValue}. Found in {end - start} seconds")
    print(f"Calculation done. Prediction for set of {len(testSet)} samples took {sum(times)} seconds.")
    plt.plot(testSet, label='Original')
    plt.plot(predictions, color='red', label='predicted')
    plt.legend(loc='best')
    plt.title(f'Prediction for {name} set')
    plt.show()


def fitArimaAndShowPacfAcf(series, order, lag):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    print(model_fit.mle_retvals)
    plot_pacf(series, lags=lag)
    plt.show()
    plot_acf(series, lags=lag)
    plt.show()


def get_stationarity(timeseries):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Dickey–Fuller test:
    result = adfuller(timeseries.values)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


def printDataset(data):
    result = [x[1] for x in data]
    plt.plot(result, label='Original')
    plt.legend(loc='best')
    plt.title(f'Dataset')
    plt.show()


def roundUpData(data):
    return [math.ceil(x) for x in data]


def autofillWithZeros(data):
    dates = pd.date_range('01.01.2007', '31.12.2020')
    dates = [[x._date_repr[8:] + '.' + x._date_repr[5:7] + '.' + x._date_repr[0:4], '0'] for x in dates]
    data = list(reversed(data))
    for data_chunk in data:
        start = 0
        for i in range(start, len(dates)):
            if data_chunk[0] == dates[i][0]:
                dates[i][1] = str(int(data_chunk[1]) + int(dates[i][1]))
                start = i
    return dates
