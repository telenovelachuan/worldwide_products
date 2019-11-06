import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


raw = pd.read_csv("../data/raw/raw.csv")
raw['Order_Demand'] = raw['Order_Demand'].apply(lambda x: int(x.replace("(", "").replace(")", "")))
# raw.to_csv("../data/processed/processed.csv")

'''
the order trend by product category and warehouse
'''


def aggregate_order_for_df(cat_df):
    date_pt = cat_df.iloc[0]['Date']
    sum_pt = int(cat_df.iloc[0]['Order_Demand'])
    result = [sum_pt]
    for idx, row in cat_df.iterrows():
        if row['Date'] != date_pt:
            print date_pt
            result.append(sum_pt)
            date_pt = row['Date']
            sum_pt = 0
        else:
            sum_pt += int(row['Order_Demand'])
    return result


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


def visualize_orders_by_attribute(attr_name):
    attr_values = list(raw[attr_name].unique())
    for attr_value in attr_values:
        df = raw[raw[attr_name] == attr_value]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date']).dropna()

        agg_orders = aggregate_order_for_df(df)
        dates = sorted(list(df['Date'].unique()))

        print "agg_orders:{}".format(agg_orders)
        fig = plt.figure(1, figsize=[14, 7])
        plt.ylabel('Orders per {}'.format(attr_name))
        plt.xlabel('Day')
        plt.title('Orders in {} {}'.format(attr_name, attr_value))
        plt.plot(dates, agg_orders)
        plt.legend()
        plt.show()


#visualize_orders_by_attribute('Product_Category')
#visualize_orders_by_attribute('Warehouse')

'''
pick some product entries by category to look at
'''


def visualize_product_trend(product_code):
    df = raw[raw.Product_Code == product_code]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()

    agg_orders = aggregate_order_for_df(df)
    dates = sorted(list(df['Date'].unique()))

    print "agg_orders:{}".format(agg_orders)
    fig = plt.figure(1, figsize=[14, 7])
    plt.ylabel('Orders')
    plt.xlabel('Day')
    plt.title('Orders of product {}'.format(product_code))
    plt.plot(dates, agg_orders)
    plt.show()


#visualize_product_trend('Product_0183')
visualize_product_trend('Product_0606')


'''
pick the most ordered product by category
'''
most_ordered_products_by_ctg = ['Product_0606', 'Product_1101', 'Product_1361']
#most_ordered_products_by_ctg = ['Product_000{}'.format(i) for i in range(1, 9)]

# for product in most_ordered_products_by_ctg:
#     visualize_product_trend(product)


'''
autocorrelations
'''
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
for product in most_ordered_products_by_ctg:
    df = raw[raw.Product_Code == product]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    agg_orders = aggregate_order_for_df(df)
    dates = sorted(list(df['Date'].unique()))

    fig = plt.figure(1, figsize=[10, 5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = np.array(agg_orders)
    print "data:{}".format(data)
    data_diff = [data[i] - data[i - 1] for i in range(1, len(data))]
    print "data_diff:{}".format(data_diff)
    autocorr = acf(data_diff)
    pac = pacf(data_diff)

    x = [x for x in range(len(pac))]
    ax1.plot(x[1:], autocorr[1:])

    ax2.plot(x[1:], pac[1:])
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title(product)

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    plt.show()


# def plot_attr_value(attr, attr_value):
#     df = raw[raw[attr] == attr_value]
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.sort_values(by=['Date']).dropna()
#
#     plot_acf(df['Order_Demand'], lags=5, title=attr_value)
#     plt.show()
#
#
# plot_attr_value('Product_Code', 'Product_0606')


'''
the order trend by month
'''


# products = ['Product_000{}'.format(i) for i in range(1, 9)]
for product in most_ordered_products_by_ctg:
    data, dates = agg_by_month('Product_Code', product)
    fig = plt.figure(1, figsize=[14, 7])
    plt.ylabel('Orders')
    # plt.xlabel('')
    plt.title('Orders of product {} by month'.format(product))
    plt.plot(dates, data)
    plt.show()

'''
shifting
'''
def plot_shifts(attr, attr_value, offset):
    df = raw[raw[attr] == attr_value]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()
    df = df.set_index('Date')
    grouper = df.groupby(pd.Grouper(freq="M"))
    sum = grouper.sum()
    df_shift = sum.shift(offset)

    sum['Order_Demand'].plot(legend=True)
    shifted = df_shift['Order_Demand'].plot(legend=True)
    shifted.legend([attr_value, '{} lagged'.format(attr_value)])
    plt.show()

# plot_shifts('Product_Code', 'Product_0606', 3)


'''
comparing different categories by month
'''
def compare_attr_by_month(attr):

    unique_values = list(raw[attr].unique())
    for value in unique_values:
        agg_data, dates = agg_by_month(attr, value, return_raw=True)
        plot = agg_data.plot(legend=True)

    plot.legend(unique_values)
    plt.show()

#compare_attr_by_month('Product_Category')
#compare_attr_by_month('Warehouse')

'''
comparing warehouse on product
'''
def compare_prod_by_warehouse(product):
    df = raw[raw['Product_Code'] == product]
    unique_values = list(df['Warehouse'].unique())
    for value in unique_values:
        agg_data, dates = agg_attrs_by_month({'Warehouse': value}, return_raw=True)
        plot = agg_data.plot(legend=True)
    plot.legend(unique_values)
    plt.title('Monthly order trends on {} by warehouses'.format(product))
    plt.show()


compare_prod_by_warehouse('Product_0243')
