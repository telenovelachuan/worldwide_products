import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from joblib import dump


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


def decompose_attr_value(attr, attr_value, basis='daily'):
    df = raw[raw[attr] == attr_value]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    df = df.set_index('Date')
    data = df['Order_Demand']
    dates = []

    if basis == 'monthly':
        data, dates = agg_attrs_by_month({attr: attr_value}, return_raw=True)

    decomposed = sm.tsa.seasonal_decompose(data, freq=len(data))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    decomposed.trend.plot(ax=ax1)
    decomposed.resid.plot(ax=ax2)
    decomposed.seasonal.plot(ax=ax3)
    # plt.title("decomposition of {}".format(attr_value))
    plt.show()

decompose_attr_value('Warehouse', 'Whse_A')
decompose_attr_value('Product_Category', 'Category_028')


'''
ARIMA
'''
# df = raw[raw['Product_Code'] == 'Product_0606']
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values(by=['Date']).dropna()
# df = df.set_index('Date')
# df = df['Order_Demand']

df, dates = agg_by_month('Product_Code', 'Product_0606', return_raw=True)

# run Augmented Dickey Fuller test to see if differencing is needed
from statsmodels.tsa.stattools import adfuller
result = adfuller(df)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# ADF Statistic: -39.97
# p-value: 0.00, can set differencing to 0


model = ARIMA(df, order=(3, 2, 1))
model_fit = model.fit()
print model_fit.summary()
model_fit.save("../models/ARIMA_3_2_1.pickle")
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
#
# predict
model_fit.plot_predict(dynamic=False)
plt.show()
# not so accurate on spikes

# find optimal model by cross-validation
index_p85 = int(len(df.index) * 0.85)
train = df[:index_p85]
test = df[index_p85:]
# Build Model
# model = ARIMA(train, order=(3,2,1))
model = ARIMA(train, order=(3, 2, 1))
fitted = model.fit(disp=-1)
print fitted.summary()

# Forecast
fc, se, conf = fitted.forecast(9, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
#
# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
# changing order from 1,1,1 to 3,2,1


def train_arima_predict_n_plot(attr_dict):
    df, dates = agg_attrs_by_month(attr_dict, return_raw=True)

    index_p85 = int(len(df.index) * 0.85)
    train = df[:index_p85]
    test = df[index_p85:]

    model = ARIMA(train, order=(3, 2, 1))
    fitted = model.fit(disp=-1)
    print fitted.summary()

    fc, se, conf = fitted.forecast(9, alpha=0.05)  # 95% conf
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


'''
Linear Regression
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def train_regression_model_n_predict(attr_dict):
    df, dates = agg_attrs_by_month(attr_dict, return_raw=True)

    index_p85 = int(len(df.index) * 0.85)
    train = df[:index_p85]
    test = df[index_p85:]

    poly_features = PolynomialFeatures(degree=5, include_bias=True)
    X_train = [[x * 2] for x in range(1, len(train.index) + 1)]
    Y_train = [[y] for y in train]
    X_poly = poly_features.fit_transform(X_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, Y_train)
    dump(lin_reg, '../models/polynomial.joblib')
    print "polynomial intercept: {}, polynomial coef: {}".format(lin_reg.intercept_, lin_reg.coef_)
    X_test = [[x * 2] for x in range(len(train.index), len(train.index) + len(test.index))]
    X_poly_test = poly_features.fit_transform(X_test)
    y_pred = lin_reg.predict(X_poly_test)

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(X_train, Y_train, label='training')

    # plt.plot(test, label='actual')
    plt.plot(X_test, y_pred, label='forecast')
    plt.title('Forecast vs Actuals on {}'.format(attr_dict))
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


train_regression_model_n_predict({'Product_Code': 'Product_0606'})


