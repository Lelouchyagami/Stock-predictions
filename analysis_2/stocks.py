from regression import regression
from test_train import test_train
from scaling import scaling
from data_preprocess import data_preprocess
from predict_plot import predict_plot
import pandas as pd

df = pd.read_csv("a.us.txt")
df.set_index('Date',inplace=True)

df = data_preprocess(df)
print(df.head())


features , labels = scaling(df)
features_train,labels_train,features_test,labels_test = test_train(features,labels)
reg = regression(features_train,features_test,labels_train,labels_test)
predict_plot(reg , features,df)
