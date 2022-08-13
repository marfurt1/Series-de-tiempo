# Import libarys
import pandas as pd
import numpy as np 

from datetime import datetime

import seaborn as sns
import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt

from pylab import rcParams
import statsmodels.api as sm

from pmdarima.arima import auto_arima

import pickle

# Load datasets

data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_test_a.index = pd.to_datetime(data_test_a.index)
data_test_b.index = pd.to_datetime(data_test_b.index)


###############

# load the model from data
filename = '../models/best_model_a.pkl'
load_model_a = pickle.load(open(filename, 'rb'))

predicciones_a=load_model_a.fit_predict(data_test_a,n_periods=60*24)

print('las predicciones para el próximo día de  test a son: {}'.format(predicciones_a))


filename = '../models/best_model_b.pkl'
load_model_b = pickle.load(open(filename, 'rb'))

# reentreno el modelo con la ventana móvil (historial)
predicciones_b=load_model_b.fit_predict(data_test_b,n_periods=60*24)

print('las predicciones para el próximo día de  test b son: {}'.format(predicciones_b))






