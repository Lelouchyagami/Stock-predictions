import pandas as pd 
import math

def data_preprocess(df):
	df2 = df[["Open","High","Low","Close","Volume"]]
	df2["HL_PCT"] = ((df2["High"]-df2["Low"])/df2["Low"])*100
	df2["PCT_Change"] = ((df2["Close"]-df2["Open"])/df2["Open"])*100

	pred_col = 'Close'
	shift = int(math.ceil(0.01*len(df2)))
	df2["label"] = df2[pred_col].shift(-shift)
	df2.dropna(inplace = True)
	return df2


