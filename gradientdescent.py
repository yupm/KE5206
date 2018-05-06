import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from neupy import algorithms, estimators, environment, layers, architectures
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

'''
hnet = algorithms.Hessian(
    connection=[
        layers.Input(58),
        layers.Sigmoid(50),
        layers.Sigmoid(10),
        layers.Sigmoid(1),
    ],
    verbose=True,
)

adanet = algorithms.Adamax(
    connection=[
        layers.Input(58),
        layers.Sigmoid(50),
        layers.Sigmoid(10),
        layers.Sigmoid(1),
    ],
    verbose=True,
)




hnet.train(x_train, y_train)

hresult = hnet.predict(x_test)

herror = estimators.rmse(hresult, y_test)






adanet.train(x_train, y_train)

aresult = adanet.predict(x_test)

aerror = estimators.rmse(aresult, y_test)
'''



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
print("Start moe training")

gdnet.train(x_train, y_train, epochs=500)

gresult = gdnet.predict(x_test)

gerror = estimators.rmse(gresult, y_test)
print(gerror)








