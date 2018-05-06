import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from neupy import algorithms, estimators, environment, layers, architectures
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

df = pd.read_csv("OnlineNewsPopularity.csv")

headers = df.columns[0:61]


#remove recent news (less than 2 months)
df = df[df[' timedelta'] > 60]

#Conduct PCA


data=df[df.columns[2:60]]


target = df[' shares'].ravel()

data_norm = StandardScaler().fit_transform(data)





network = architectures.mixture_of_experts([
     layers.join(
         layers.Input(58),
         layers.Softmax(22),
         layers.Softmax(1),
     ),
     layers.join(
         layers.Input(58),
         layers.Relu(60),
         layers.Relu(40),
         layers.Softmax(22),
         layers.Softmax(1),
     ),
     layers.join(
         layers.Input(58),
         layers.Tanh(12),
         layers.Tanh(25),
         layers.Tanh(1),
     ),
])
network
gdnet = algorithms.Adam(network, verbose=True)
gdnet.fit(data_norm,target, epochs=500)

predicted = cross_val_predict(gdnet, data_norm, target, cv=5)



error = estimators.rmse(target, predicted)

print("MOE RMSE = {}\n".format(error))

r2_score = metrics.r2_score(target, predicted)

print("MOE R_SCORE = {}\n".format(r2_score))


