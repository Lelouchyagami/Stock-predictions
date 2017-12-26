import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import datetime
from matplotlib import style
style.use('ggplot')

def predict_plot(clf,X,df):
	forecast_out = int(math.ceil(0.01*len(df)))
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]

	forecast_set = clf.predict(X_lately)
	df2 = pd.DataFrame(forecast_set)

	df['Close'].plot()
	plt.title('original predictions')
	df2.plot()
	plt.title('future predictions')
	plt.legend(loc=4)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.show()	