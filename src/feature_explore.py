import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


raw = pd.read_csv("../data/raw/raw.csv")

'''
the order trend by product category
'''


def aggregate_order_for_cat(cat_df, dates):
    result = []
    for date in dates:
        result.append(sum([int(o['Order_Demand'].replace("(", "").replace(")", "")) for _, o in cat_df.iterrows() if datetime.strptime(o['Date'], "%Y/%m/%d").date() <= date]))
    return result


categories = list(raw['Product_Category'].unique())
for category in categories:
    df = raw[raw.Product_Category == category].sort_values(by=['Date']).dropna()
    dates = [datetime.strptime(d, "%Y/%m/%d").date() for d in list(df['Date'].unique())]
    agg_orders = aggregate_order_for_cat(df, dates)
    print "agg_orders:{}".format(agg_orders)
    fig = plt.figure(1, figsize=[10, 10])
    plt.ylabel('Orders per category')
    plt.xlabel('Day')
    plt.title('Orders in Different Categories')
    plt.plot(dates, agg_orders)





