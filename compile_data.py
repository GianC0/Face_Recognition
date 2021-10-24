import pandas as pd
from pathlib import Path

p = Path("results/resultsAll.txt")

data = pd.read_csv(p, sep=" ")

data_nn = data[data['NN'] == "Net"]
data_nn_e = data[data['NN'] == "NetExtraLinear"]

data_nn_5 = data_nn[data_nn['Epochs']==5]
data_nn_10 = data_nn[data_nn['Epochs']==10]
data_nn_15 = data_nn[data_nn['Epochs']==15]
data_nn_20 = data_nn[data_nn['Epochs']==20]
data_nn_25 = data_nn[data_nn['Epochs']==25]

data_nn_e_5 = data_nn_e[data_nn_e['Epochs']==5]
data_nn_e_10 = data_nn_e[data_nn_e['Epochs']==10]
data_nn_e_15 = data_nn_e[data_nn_e['Epochs']==15]
data_nn_e_20 = data_nn_e[data_nn_e['Epochs']==20]
data_nn_e_25 = data_nn_e[data_nn_e['Epochs']==25]

print(data_nn_5["Accuracy"].mean())
print(data_nn_10["Accuracy"].mean())
print(data_nn_15["Accuracy"].mean())
print(data_nn_20["Accuracy"].mean())
print(data_nn_25["Accuracy"].mean())
print()
print(data_nn_5["time_test"].mean())
print(data_nn_10["time_test"].mean())
print(data_nn_15["time_test"].mean())
print(data_nn_20["time_test"].mean())
print(data_nn_25["time_test"].mean())
print()
print(data_nn_e_5["Accuracy"].mean())
print(data_nn_e_10["Accuracy"].mean())
print(data_nn_e_15["Accuracy"].mean())
print(data_nn_e_20["Accuracy"].mean())
print(data_nn_e_25["Accuracy"].mean())
print()
print(data_nn_e_5["time_test"].mean())
print(data_nn_e_10["time_test"].mean())
print(data_nn_e_15["time_test"].mean())
print(data_nn_e_20["time_test"].mean())
print(data_nn_e_25["time_test"].mean())
