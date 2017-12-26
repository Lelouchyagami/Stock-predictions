import numpy as np 
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression


def regression(features_train,labels_train,features_test,labels_test):
	reg = LinearRegression()	
	reg.fit(features_train,labels_train)
	print("the reg coefficients are:")
	print(reg.coef_)
	print("The intercept is :")
	print(reg.intercept_)
	print("The accuracy is:")
	print(reg.score(features_test,labels_test)*100)
	return reg
