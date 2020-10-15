import numpy as np
import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import graphviz
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = np.load('train.npz') 

x = data['x'] #feature matrix
print(x.shape)
x_re = np.reshape(x, (72000, 392))
y = data['y'] #label matrix
location = data['locations'] #location matrix
times = data['times'] #time matrix

data_val = np.load("val.npz")

val_x = data_val['x'] #feature matrix
print(np.size(val_x, 0))
val_x_re = np.reshape(val_x, (np.size(val_x, 0), 392))
val_y = data_val['y'] #label matrix
val_location = data_val['locations'] #location matrix
val_times = data_val['times'] #time matrix

# 1 baseline
dic_region = {}

for i in range(10):
	for j in range(10):
		dic_region[(i, j)] = [0, 0]

for i in range(np.size(y, 0)):
	# print(y[i][0])
	dic_region[(location[i][0], location[i][1])][0] += y[i][0]
	dic_region[(location[i][0], location[i][1])][1] += 1

# print(dic_region)

baseline_pred = np.array([dic_region[(loc[0], loc[1])][0]/dic_region[(loc[0], loc[1])][1] for loc in val_location])
print(mean_squared_error(val_y, baseline_pred, squared=False))

# 2 linear regression
reg = LinearRegression().fit(x_re, y)
y_pred = reg.predict(val_x_re)
print(mean_squared_error(val_y, y_pred, squared=False))

# 3 xgboost
def xgb_parameter_tune(train_val_X, train_val_y):
	rates = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
	train_rmse_all = []
	val_rmse_all = []
	
	for rate in rates:
		##########################
		# learning_rate
		gbm = xgb.XGBClassifier(learning_rate=rate, random_state=1)
		gbm.fit(train_val_X, train_val_y)
		
		y_pred_train = gbm.predict(train_val_X)
		y_pred_val = gbm.predict(val_x_re)
		
		train_rmse = mean_squared_error(train_val_y, y_pred_train, squared=False)
		val_rmse = mean_squared_error(val_y, y_pred_val, squared=False)
		##########################

		print("rate: ", rate)
		print("Training RMSE: ", train_rmse)
		print("Validation RMSE: ", val_rmse)

		train_rmse_all.append(train_rmse)
		val_rmse_all.append(val_rmse)

	return rates, val_rmse_all

print("\\\\\\")
print("xgboost")

rates, val_rmse_all = xgb_parameter_tune(x_re, y)
print("mean: ", np.mean(val_rmse_all))
print("std: ", np.std(val_rmse_all))
