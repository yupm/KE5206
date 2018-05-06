import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from neupy import algorithms, estimators, environment, layers
from sklearn import metrics

df = pd.read_csv("OnlineNewsPopularity.csv")

headers = df.columns[0:61]


#remove recent news (less than 2 months)
df = df[df[' timedelta'] > 60]

#Conduct PCA


data=df[df.columns[2:60]]


target = df[' shares'].ravel()

data_norm = StandardScaler().fit_transform(data)



x_train, x_test, y_train, y_test = train_test_split(data_norm, target, test_size=0.3)


pnn_network = algorithms.PNN(std=0.1, verbose=True)
pnn_network.train(x_train, y_train)

result = pnn_network.predict(x_test)


error = estimators.rmse(result, y_test)

print("GRNN RMSE = {}\n".format(error))

r2_score = metrics.r2_score(result, y_test)

print("GRNN R_SCORE = {}\n".format(r2_score))


