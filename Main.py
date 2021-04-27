import csv
import LSTM, MLP


import Helpers

with open('data/10x14f.csv') as f:
    reader = csv.reader(f)
    _10x14f = list(reader)

with open('data/15x30l.csv') as f:
    reader = csv.reader(f)
    _15x30l = list(reader)

with open('data/GP-4x.csv') as f:
    reader = csv.reader(f)
    _GP4x = list(reader)

with open('data/GP-6x.csv') as f:
    reader = csv.reader(f)
    _GP6x = list(reader)

with open('data/pwz52.csv') as f:
    reader = csv.reader(f)
    _pwz52 = list(reader)

with open('data/szybka.csv') as f:
    reader = csv.reader(f)
    _szybka = list(reader)

_10x14f_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_10x14f)
_15x30l_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_15x30l)
_GP4x_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_GP4x)
_GP6x_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_GP6x)
_pwz52_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_pwz52)
_szybka_final = Helpers.preprocessDataForTwoColumnsAndNoZeros(_szybka)

_GP6x_day = Helpers.autofillWithZeros(_GP6x_final)

_10x14f_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_10x14f_final))))
_15x30l_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_15x30l_final))))[12:]
_GP4x_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_GP4x_final))))[12:]
_GP6x_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_GP6x_final))))
_pwz52_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_pwz52_final))))[12:]
_szybka_month = Helpers.addMissingData(list(reversed(Helpers.countAmountPerMonth(_szybka_final))))[12:]

series = Helpers.transformDataIntoSeries(_GP6x_month)

order = (36, 0, 24)
type = 'short'
lag = 60
name = 'pwz52'
onlyshow = 1
arima = 0

if arima:
    if onlyshow:
        Helpers.fitArimaAndShowPacfAcf(series, order, lag)
        Helpers.get_stationarity(series)
        df_log_shift = series - series.shift()
        df_log_shift.dropna(inplace=True)
        Helpers.get_stationarity(df_log_shift)
    else:
        Helpers.arimaModel(series, order, name, type)
else:
    # Helpers.printDataset(_GP6x_month)

    # mlp = MLP.MLPNN('GP6x', _GP6x_month, 17, 1900)
    # mlp.single_MLP()

    lstm = LSTM.LSTMNN('GP6x', _GP6x_month, 10, 5000)
    lstm.single_LSTM()
