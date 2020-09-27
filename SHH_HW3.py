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

data = pd.read_csv("credit_card_train.csv")

array = data.values
X = array[:,0:30]
y = array[:,30]
random.seed(0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# 1 dummy
def dummy_parameter(train_val_X, train_val_y):
	strategies = ["most_frequent"]
	
	kf = StratifiedKFold(n_splits = 5)
	for strategy in strategies:
		train_auc = []
		val_auc = []
		
		for train_index, val_index in kf.split(train_val_X, train_val_y):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
			# dummy most_frequent
			dummy_clf = DummyClassifier(strategy=strategy)
			dummy_clf.fit(train_X, train_y)
			
			y_pred_train = dummy_clf.predict_proba(train_X)
			y_pred_val = dummy_clf.predict_proba(val_X)
			
			train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
			val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))

			# train_auc.append(dummy_clf.score(train_X, train_y))
			# val_auc.append(dummy_clf.score(val_X, val_y))

			print("N_th fold train AUC", roc_auc_score(train_y, y_pred_train[:, 1]))
			print("N_th fold val AUC", roc_auc_score(val_y, y_pred_val[:, 1]))
			##########################
		
		avg_train_auc = sum(train_auc) / len(train_auc)
		avg_val_auc = sum(val_auc) / len(val_auc)
		print("strategies: ", strategy)
		print("Training AUC: ", avg_train_auc)
		print("Validation AUC: ", avg_val_auc)

	return 
# dummy_parameter(X, y)
	

# 2 random forest
def rf_parameter_tune(train_val_X, train_val_y):
	n_estimators = [1, 3, 5, 10, 15, 20, 30, 50]

	train_auc_all = []
	val_auc_all = []
	
	kf = StratifiedKFold(n_splits = 5)
	for estimator in n_estimators:
		train_auc = []
		val_auc = []
		
		for train_index, val_index in kf.split(train_val_X, train_val_y):
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]	
			
			sm = SMOTE(random_state=42)
			X_res, y_res = sm.fit_resample(train_X, train_y)
			
			rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1, random_state=1)
			rf.fit(X_res, y_res)
			
			y_pred_train = rf.predict_proba(train_X)
			y_pred_val = rf.predict_proba(val_X)

			train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
			val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))
			##########################
		
		avg_train_auc = sum(train_auc) / len(train_auc)
		avg_val_auc = sum(val_auc) / len(val_auc)
		print("estimators: ", estimator)
		print("Training AUC: ", avg_train_auc)
		print("Validation AUC: ", avg_val_auc)

		train_auc_all.append(avg_train_auc)
		val_auc_all.append(avg_val_auc)
	
	return n_estimators, val_auc_all

n_estimators, val_auc_all = rf_parameter_tune(X, y)
print("\\\\\\")
print("random forest")
print("mean: ", np.mean(val_auc_all))
print("std: ", np.std(val_auc_all))

# plt.plot(n_estimators, train_auc_all, marker='.', label="Training AUC")
# plt.plot(n_estimators, val_auc_all, marker='.', label="Validation AUC")
# plt.xlabel('n_estimator')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()


# 2 xgboost
def xgb_parameter_tune(train_val_X, train_val_y):
	rates = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
	train_auc_all = []
	val_auc_all = []
	
	kf = StratifiedKFold(n_splits = 5)
	for rate in rates:
		train_auc = []
		val_auc = []
		
		for train_index, val_index in kf.split(train_val_X, train_val_y):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
			
			sm = SMOTE(random_state=42)
			X_res, y_res = sm.fit_resample(train_X, train_y)

			# learning_rate	            
			gbm = xgb.XGBClassifier(learning_rate=rate, random_state=1)
			gbm.fit(X_res, y_res)
			
			y_pred_train = gbm.predict_proba(train_X)
			y_pred_val = gbm.predict_proba(val_X)
			
			train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
			val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))
			##########################

		avg_train_auc = sum(train_auc) / len(train_auc)
		avg_val_auc = sum(val_auc) / len(val_auc)
		print("rate: ", rate)
		print("Training AUC: ", avg_train_auc)
		print("Validation AUC: ", avg_val_auc)

		train_auc_all.append(avg_train_auc)
		val_auc_all.append(avg_val_auc)

	return rates, val_auc_all

rates, val_auc_all = xgb_parameter_tune(X, y)
print("\\\\\\")
print("xgboost")
print("mean: ", np.mean(val_auc_all))
print("std: ", np.std(val_auc_all))

# plt.plot(rates, train_auc_all, marker='.', label="Training AUC")
# plt.plot(rates, val_auc_all, marker='.', label="Validation AUC")
# plt.xlabel('learning rate')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()

# 2 SVM
def svm_parameter_tune(train_val_X, train_val_y):
	C_penalty = [50, 10 , 1, 0.1, 0.01]
	train_auc_all = []
	val_auc_all = []
	
	kf = StratifiedKFold(n_splits = 5)
	for c in C_penalty:
		train_auc = []
		val_auc = []
		
		for train_index, val_index in kf.split(train_val_X, train_val_y):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
			
			sm = SMOTE(random_state=42)
			X_res, y_res = sm.fit_resample(train_X, train_y)

			# learning_rate	            
			svm = SVC(C=c, probability=True, random_state=1)
			svm.fit(X_res, y_res)
			
			y_pred_train = svm.predict_proba(train_X)
			y_pred_val = svm.predict_proba(val_X)
			
			train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
			val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))
			##########################

		avg_train_auc = sum(train_auc) / len(train_auc)
		avg_val_auc = sum(val_auc) / len(val_auc)
		print("C_penalty: ", c)
		print("Training AUC: ", avg_train_auc)
		print("Validation AUC: ", avg_val_auc)

		train_auc_all.append(avg_train_auc)
		val_auc_all.append(avg_val_auc)

	return C_penalty, val_auc_all

C_penalty, val_auc_all = svm_parameter_tune(X, y)
print("\\\\\\")
print("SVM")
print("mean: ", np.mean(val_auc_all))
print("std: ", np.std(val_auc_all))

# plt.plot(rates, train_auc_all, marker='.', label="Training AUC")
# plt.plot(C_penalty, val_auc_all, marker='.', label="Validation AUC")
# plt.xlabel('C_penalty')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()

# 2 knn
def knn_parameter_tune(train_val_X, train_val_y):
	n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15]
	train_auc_all = []
	val_auc_all = []
	
	kf = StratifiedKFold(n_splits = 5)
	for n in n_neighbors:
		train_auc = []
		val_auc = []
		
		for train_index, val_index in kf.split(train_val_X, train_val_y):
			##########################
			train_X = train_val_X[train_index,:]
			val_X = train_val_X[val_index,:]

			train_y = train_val_y[train_index]
			val_y = train_val_y[val_index]
			
			sm = SMOTE(random_state=42)
			X_res, y_res = sm.fit_resample(train_X, train_y)

			# learning_rate	            
			knn = KNeighborsClassifier(n_neighbors=n)
			knn.fit(X_res, y_res)
			
			y_pred_train = knn.predict_proba(train_X)
			y_pred_val = knn.predict_proba(val_X)
			
			train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
			val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))
			##########################

		avg_train_auc = sum(train_auc) / len(train_auc)
		avg_val_auc = sum(val_auc) / len(val_auc)
		print("n_neighbors: ", n)
		print("Training AUC: ", avg_train_auc)
		print("Validation AUC: ", avg_val_auc)

		train_auc_all.append(avg_train_auc)
		val_auc_all.append(avg_val_auc)

	return n_neighbors, val_auc_all

n_neighbors, val_auc_all = knn_parameter_tune(X, y)
print("\\\\\\")
print("knn")
print("mean: ", np.mean(val_auc_all))
print("std: ", np.std(val_auc_all))

# plt.plot(rates, train_auc_all, marker='.', label="Training AUC")
# plt.plot(n_neighbors, val_auc_all, marker='.', label="Validation AUC")
# plt.xlabel('n_neighbors')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()

# 2 naive_bayes
def NB(train_val_X, train_val_y):
	
	kf = StratifiedKFold(n_splits = 5)
	train_auc = []
	val_auc = []
	
	for train_index, val_index in kf.split(train_val_X, train_val_y):
		##########################
		train_X = train_val_X[train_index,:]
		val_X = train_val_X[val_index,:]

		train_y = train_val_y[train_index]
		val_y = train_val_y[val_index]

		sm = SMOTE(random_state=42)
		X_res, y_res = sm.fit_resample(train_X, train_y)

		gnb = GaussianNB()
		gnb.fit(X_res, y_res)
		
		y_pred_train = gnb.predict_proba(train_X)
		y_pred_val = gnb.predict_proba(val_X)
		
		train_auc.append(roc_auc_score(train_y, y_pred_train[:, 1]))
		val_auc.append(roc_auc_score(val_y, y_pred_val[:, 1]))

		# train_auc.append(dummy_clf.score(train_X, train_y))
		# val_auc.append(dummy_clf.score(val_X, val_y))

		print("N_th fold train AUC", roc_auc_score(train_y, y_pred_train[:, 1]))
		print("N_th fold val AUC", roc_auc_score(val_y, y_pred_val[:, 1]))
		##########################
	
	avg_train_auc = sum(train_auc) / len(train_auc)
	avg_val_auc = sum(val_auc) / len(val_auc)
	print("NB")
	print("Training AUC: ", avg_train_auc)
	print("Validation AUC: ", avg_val_auc)

	return
print("\\\\\\")
NB(X, y)

# # 5 test on data
# def xgb_parameter_tune(train_val_X, train_val_y):
# 	rates = [0.1]
# 	train_acc_all = []
# 	val_acc_all = []
	
# 	kf = KFold(n_splits = 5)
# 	for rate in rates:
# 		train_acc = []
# 		val_acc = []
		
# 		for train_index, val_index in kf.split(train_val_X):
# 			train_X = train_val_X[train_index,:]
# 			val_X = train_val_X[val_index,:]

# 			train_y = train_val_y[train_index]
# 			val_y = train_val_y[val_index]
	            
# 			gbm = xgb.XGBClassifier(learning_rate=rate, max_depth=15, n_estimators=500, min_child_weight=3, random_state=1)
# 			gbm.fit(train_X, train_y)
# 			train_acc.append(gbm.score(train_X, train_y))
# 			val_acc.append(gbm.score(val_X, val_y))
		
# 		avg_train_acc = sum(train_acc) / len(train_acc)
# 		avg_val_acc = sum(val_acc) / len(val_acc)
# 		print("rate: ", rate)
# 		print("Training accuracy: ", avg_train_acc * 100, "%")
# 		print("Validation accuracy: ", avg_val_acc * 100, "%")

# 		train_acc_all.append(avg_train_acc)
# 		val_acc_all.append(avg_val_acc)

# 	return rates, train_acc_all, val_acc_all

# rates, train_acc_all, val_acc_all = xgb_parameter_tune(X_train_tfidf, train_val_y)

# print("mean: ", np.mean(val_acc_all))
# print("std: ", np.std(val_acc_all))
# gbm = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, n_estimators=500, min_child_weight=3, random_state=1)

# data = pd.read_csv("news-test.csv")

# unlabeled = []

# for idx, r in data.iterrows():
# 	unlabeled.append(' '.join(str(elem) for elem in get_tokens(r["Text"])))


# train_X_final = category["sport"] + category["business"] + category["politics"] + category["entertainment"] + category["tech"]
# train_y_final = np.concatenate((s_y, b_y, p_y, e_y, t_y))

# final_X_tfidf = tfidf_vectorizer.fit_transform(train_X_final)

# unlabeled_tfidf = tfidf_vectorizer.transform(unlabeled)

# gbm.fit(final_X_tfidf, train_y_final)

# predictions = gbm.predict(unlabeled_tfidf)

# ans = {}

# for idx, r in data.iterrows():
# 	predict = ""

# 	if predictions[idx] == 0:
# 		predict = "sport"
# 	elif predictions[idx] == 1:
# 		predict = "business"
# 	elif predictions[idx] == 2:
# 		predict = "politics"
# 	elif predictions[idx] == 3:
# 		predict = "entertainment"
# 	else:
# 		predict = "tech"

# 	ans[r["ArticleId"]] = predict

# print(ans)

# pd.DataFrame.from_dict(data=ans, orient='index').to_csv('labels.csv', header=False)