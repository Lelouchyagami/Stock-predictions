import numpy as np 
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression

def test_train(features,labels):
	X_train,X_test,y_train,y_test = cross_validation.train_test_split(features,labels,test_size = 0.2)
	return X_train,X_test,y_train,y_test

