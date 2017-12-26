import numpy as np 
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression


def scaling(df):
	features  = np.array(df.drop(['label'],1))
	labels = np.array(df['label'])
	features = preprocessing.scale(features)
	#print(len(X),len(y))
	return features , labels