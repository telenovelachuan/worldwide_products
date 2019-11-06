import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


'''
Decomposition
'''
raw = pd.read_csv("../data/processed/processed.csv")
# raw['Order_Demand'] = raw['Order_Demand'].apply(lambda x: int(x.replace("(", "").replace(")", "")))


def decompose_attr_value(attr, attr_value):
    df = raw[raw[attr] == attr_value]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    df = df.set_index('Date')
    decomposed = sm.tsa.seasonal_decompose(df['Order_Demand'], freq=360)
    figure = decomposed.plot()
    # plt.title("decomposition of {}".format(attr_value))
    plt.show()

# decompose_attr_value('Warehouse', 'Whse_A')
# decompose_attr_value('Product_Category', 'Category_028')


'''
ARIMA
'''


df = raw[raw['Warehouse'] == 'Whse_J']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date']).dropna()
df = df.set_index('Date')
df = df['Order_Demand']

# run Augmented Dickey Fuller test to see if differencing is needed
from statsmodels.tsa.stattools import adfuller
result = adfuller(df)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])




# model = ARIMA(df, order=(5, 1, 0))
# model_fit = model.fit()
# print model_fit.summary()
# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())
#
# # predict
# size = int(len(df.index) * 0.8)
# train_set, test_set = df[0: size], df[size: len(df.index)]
# history = [x for x in train_set]
# predictions = []
# for t in range(len(test_set.index)):
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test_set[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
#
# error = mean_squared_error(test_set, predictions)
# print 'Test MSE: {}'.format(error)
#
# plt.plot(test_set)
# plt.plot(predictions, color='red')
# plt.show()

