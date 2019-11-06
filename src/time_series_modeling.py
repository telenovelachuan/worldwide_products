import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


'''
Decomposition
'''
raw = pd.read_csv("../data/processed/processed.csv")
# raw['Order_Demand'] = raw['Order_Demand'].apply(lambda x: int(x.replace("(", "").replace(")", "")))


def agg_by_month(attr, attr_value, return_raw=False):
    df = raw[raw[attr] == attr_value]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    df = df.set_index('Date')
    df.groupby(pd.Grouper(freq="M"))
    sum = df.groupby(pd.Grouper(freq="M")).sum()
    data = list(sum['Order_Demand'])
    dates = list(sum.index)
    if return_raw is True:
        return sum['Order_Demand'], dates
    return data, dates


def agg_attrs_by_month(attr_dict, return_raw=False):
    df = raw
    for attr in attr_dict:
        df = df[df[attr] == attr_dict[attr]]

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    df = df.set_index('Date')
    df.groupby(pd.Grouper(freq="M"))
    sum = df.groupby(pd.Grouper(freq="M")).sum()
    data = list(sum['Order_Demand'])
    dates = list(sum.index)
    if return_raw is True:
        return sum['Order_Demand'], dates
    return data, dates


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


# df = raw[raw['Product_Code'] == 'Product_0606']
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values(by=['Date']).dropna()
# df = df.set_index('Date')
# df = df['Order_Demand']

df, dates = agg_by_month('Product_Code', 'Product_0606', return_raw=True)

# # run Augmented Dickey Fuller test to see if differencing is needed
# from statsmodels.tsa.stattools import adfuller
# result = adfuller(df)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# # ADF Statistic: -39.97
# # p-value: 0.00, can set differencing to 0


# model = ARIMA(df, order=(2, 1, 1))
# model_fit = model.fit()
# print model_fit.summary()
# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1, 2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()
#
# # predict
# model_fit.plot_predict(dynamic=False)
# plt.show()
# # not so accurate on spikes

# # find optimal model by cross-validation
# index_p85 = int(len(df.index) * 0.85)
# train = df[:index_p85]
# test = df[index_p85:]
# # Build Model
# # model = ARIMA(train, order=(3,2,1))
# model = ARIMA(train, order=(3, 2, 1))
# fitted = model.fit(disp=-1)
# print fitted.summary()

# # Forecast
# fc, se, conf = fitted.forecast(9, alpha=0.05)  # 95% conf
#
# # Make as pandas series
# fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)
#
# # Plot
# plt.figure(figsize=(12, 5), dpi=100)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()
# # changing order from 1,1,1 to 3,2,1

import scipy