import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

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

_10x14f_month = list(reversed(Helpers.countAmountPerMonth(_10x14f_final)))
_15x30l_month = list(reversed(Helpers.countAmountPerMonth(_15x30l_final)))
_GP4x_month = list(reversed(Helpers.countAmountPerMonth(_GP4x_final)))
_GP6x_month = list(reversed(Helpers.countAmountPerMonth(_GP6x_final)))
_pwz52_month = list(reversed(Helpers.countAmountPerMonth(_pwz52_final)))
_szybka_month = list(reversed(Helpers.countAmountPerMonth(_szybka_final)))

chart = np.array(_szybka_month)
values = chart[:, 1].astype(int)
series = Helpers.changeColumnIntoSeries(values)

order = (30, 0, 9)
type = 'long'
lag = 60
name = 'pwz52'
onlyshow = 0

if onlyshow:
    Helpers.fitArimaAndShowPacfAcf(series, order, lag)
    Helpers.get_stationarity(series)
    df_log_shift = series - series.shift()
    df_log_shift.dropna(inplace=True)
    Helpers.get_stationarity(df_log_shift)
else:
    Helpers.arimaModel(series, order, name, type)
