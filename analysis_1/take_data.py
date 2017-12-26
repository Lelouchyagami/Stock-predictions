import pandas as pd 
import math
import numpy as np 
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


df = pd.read_csv("a.us.txt")

df2 = df[["Open","High","Low","Close","Volume"]]
df2["HL_PCT"] = ((df2["High"]-df2["Low"])/df2["Low"])*100
df2["PCT_Change"] = ((df2["Close"]-df2["Open"])/df2["Open"])*100

pred_col = 'Close'
shift = int(math.ceil(0.01*len(df2)))
df2["label"] = df2[pred_col].shift(-shift)
df2.dropna(inplace = True)

def preprocess(df):
	features  = np.array(df.drop(['label'],1))
	labels = np.array(df['label'])
	features = preprocessing.scale(features)
	#print(len(X),len(y))
	return features , labels
X , y = preprocess(df2)

def test_train(features,labels):
	X_train,X_test,y_train,y_test = cross_validation.train_test_split(features,labels,test_size = 0.2)
	return X_train,X_test,y_train,y_test
features_train , features_test , labels_train , labels_test = test_train(X,y)

reg = LinearRegression()
reg.fit(features_train,labels_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(features_test,labels_test))

print(len(features_train[:,0]))
plt.scatter(features_train[:,0] , labels_train , color = 'b')
plt.scatter(features_train[:,1] , labels_train , color = 'g')
plt.scatter(features_train[:,2] , labels_train , color = 'r')
plt.scatter(features_train[:,3] , labels_train , color = 'c')
plt.scatter(features_train[:,4] , labels_train , color = 'm')
plt.scatter(features_train[:,5] , labels_train , color = 'y')
plt.scatter(features_train[:,6] , labels_train , color = 'k')
plt.xlabel("features")
plt.ylabel("label")
plt.show()

plt.plot( features_test, reg.predict(features_test) )
plt.show()