import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


raw = pd.read_csv("../data/raw/raw.csv")

'''
the order trend by product category
'''


def aggregate_order_for_cat(cat_df):
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
            sum_pt += int(row['Order_Demand'].replace("(", "").replace(")", ""))
    return result


categories = list(raw['Product_Category'].unique())
for category in categories:
    df = raw[raw.Product_Category == category]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).dropna()

    agg_orders = aggregate_order_for_cat(df)
    dates = sorted(list(df['Date'].unique()))

    print "agg_orders:{}".format(agg_orders)
    fig = plt.figure(1, figsize=[14, 8])
    plt.ylabel('Orders per category')
    plt.xlabel('Day')
    plt.title('Orders in Category {}'.format(category))
    plt.plot(dates, agg_orders)

plt.show()





