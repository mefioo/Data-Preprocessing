import csv
import MLP


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

_10x14f_month = Helpers.addMissingData(_10x14f_month)
_15x30l_month = Helpers.addMissingData(_15x30l_month)[12:]
_GP4x_month = Helpers.addMissingData(_GP4x_month)[12:]
_GP6x_month = Helpers.addMissingData(_GP6x_month)
_pwz52_month = Helpers.addMissingData(_pwz52_month)[12:]
_szybka_month = Helpers.addMissingData(_szybka_month)[12:]

# print(f"10x14f {len(_10x14f_month)/12}")  #13
# print(f"15x30l {len(_15x30l_month)/12}")  #14
# print(f"GP4x {len(_GP4x_month)/12}")      #14
# print(f"GP6x {len(_GP6x_month)/12}")      #13
# print(f"pwz52 {len(_pwz52_month)/12}")    #14
# print(f"szybka {len(_szybka_month)/12}")  #14

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
    neurons = [x for x in range(5, 15)]
    epochs = [x for x in range(200, 1000, 100)]
    parameters = [[5, 400], [6, 700], [8, 300], [9, 400], [9, 500], [11, 700], [12, 400], [12, 600], [13, 900]]
    parameters = [[12, 400] for i in range(10)]

    mlp = MLP.MLPNN(_GP6x_month, 1100, 15)
    mlp.single_MLP()

    # Helpers.printDataset(_GP6x_month)

    # MLP.MLP(_szybka_month, epochs, neurons)
    # MLP.compareMLPs(_pwz52_month, parameters)
    # MLP.singleMLP(_GP6x_month, 1100, 15)

    # MLP.shortMLP(_GP6x_month, 1000, 250)


### _GP4x_month
# Epochs: 50, mean: 14.343587862245823
# Epochs: 100, mean: 13.37321615602265
# Epochs: 250, mean: 12.79097922424582
# Epochs: 500, mean: 14.466035771177559
# Epochs: 1000, mean: 14.107073291684292
# Epochs: 2000, mean: 14.658024100122471
# Epochs: 5000, mean: 14.110245799691338

# Neurons: 1, mean: 13.733801841663995
# Neurons: 2, mean: 13.50375260822307
# Neurons: 3, mean: 12.964490457169637
# Neurons: 4, mean: 13.114200505181998
# Neurons: 5, mean: 12.772144609319255
# Neurons: 6, mean: 12.956626280670303
# Neurons: 7, mean: 13.054019954807284
# Neurons: 8, mean: 13.273711817326788
# Neurons: 9, mean: 13.230705484555415
# Neurons: 10, mean: 13.671764578181044
# Neurons: 11, mean: 13.76162733982827
# Neurons: 12, mean: 13.442643765943206


### _GPx6_month
# Epochs: 50, mean: 41.970510305024476
# Epochs: 100, mean: 37.26827266332147
# Epochs: 250, mean: 37.438650926292965
# Epochs: 500, mean: 34.90812512587372
# Epochs: 1000, mean: 36.30285889548877
# Epochs: 2000, mean: 38.27564636672984
# Epochs: 50, mean: 0.45659690827311844
# Epochs: 100, mean: 0.44927393755618084
# Epochs: 250, mean: 0.4513894527838346
# Epochs: 500, mean: 0.4770499726002786
# Epochs: 1000, mean: 0.4938849858006121
# Epochs: 2000, mean: 0.48377348129286013

### _10x14f_month
# Epochs: 50, mean: 93.51938630338424
# Epochs: 100, mean: 95.45290549496318
# Epochs: 250, mean: 97.7924797324386
# Epochs: 500, mean: 100.28296844422135
# Epochs: 1000, mean: 99.11416945366099
# Epochs: 2000, mean: 99.07945478936259

### _15x30l_month
# Epochs: 50, mean: 31.30542915470887
# Epochs: 100, mean: 30.651375696769765
# Epochs: 250, mean: 30.406492551923357
# Epochs: 500, mean: 29.88787178398521
# Epochs: 1000, mean: 30.34905020244729
# Epochs: 2000, mean: 29.721777421386058

### _pwz52_month
# Epochs: 50, mean: 6.540431977532736
# Epochs: 100, mean: 5.865759843778302
# Epochs: 250, mean: 6.013568710946163
# Epochs: 500, mean: 6.486333138860043
# Epochs: 1000, mean: 6.6032075817266165
# Epochs: 2000, mean: 6.787139728527462

### _szybka_month
# Epochs: 50, mean: 23.55558577750018
# Epochs: 100, mean: 22.15059723662811
# Epochs: 250, mean: 25.703434784789277
# Epochs: 500, mean: 28.633142258057006
# Epochs: 1000, mean: 36.299863055082916
# Epochs: 2000, mean: 45.23911320924894