import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

facebook_trend = pd.read_csv("summary_stocks_trends/trend_stock_facebook.csv")
amazon_trend = pd.read_csv("summary_stocks_trends/trend_stock_amazon.csv")
apple_trend = pd.read_csv("summary_stocks_trends/trend_stock_apple.csv")
google_trend = pd.read_csv("summary_stocks_trends/trend_stock_google.csv")
tesla_trend = pd.read_csv("summary_stocks_trends/trend_stock_tesla.csv")
netflix_trend = pd.read_csv("summary_stocks_trends/trend_stock_netflix.csv")

facebook_market_data = pd.read_csv("stock reader/stock_market_data-FB.csv")
amazon_market_data = pd.read_csv("stock reader/stock_market_data-AMZN.csv")
apple_market_data = pd.read_csv("stock reader/stock_market_data-AAPL.csv")
google_market_data = pd.read_csv("stock reader/stock_market_data-GOOG.csv")
tesla_market_data = pd.read_csv("stock reader/stock_market_data-TSLA.csv")
netflix_market_data = pd.read_csv("stock reader/stock_market_data-NFLX.csv")

stocks_names_list = ["facebook", "amazon", "apple", "google", "tesla", "netflix"]

list_stocks_trend = [facebook_trend, amazon_trend, apple_trend,
                     google_trend, tesla_trend, netflix_trend]

list_stocks_market = [facebook_market_data, amazon_market_data, apple_market_data,
                      google_market_data, tesla_market_data, netflix_market_data]
list_value_close = []
for item in list_stocks_market:
    item = item.sort_values(by=['Date'])
    list_value_close.append([item['Close']])

list_of_lists_positive = []
list_of_lists_neg = []
list_of_lists_uncertain = []

for item in list_stocks_trend:
    pos = item[(item['trend'] == 'positive')]
    neg = item[(item['trend'] == 'negative')]
    uncertainty = item[(item['trend'] == 'uncertainty')]
    list_pos = list(pos['count_trend'])
    list_neg = list(neg['count_trend'])
    list_uncertainty = list(uncertainty['count_trend'])
    del list_pos[4:6] # not trading dates
    del list_neg[4:6] # not trading dates
    del list_uncertainty[4:6] # not trading dates
    list_of_lists_positive.append(list_pos)
    list_of_lists_neg.append(list_neg)
    list_of_lists_uncertain.append(list_uncertainty)


def calculate_pearson(data_1, data_2):
  """
  calculates the correlation using pearson formula
  :param data_1: the first list of values
  :param data_2: the second list of values
  :return:
  """
  corr, _ = pearsonr(data_1, data_2)
  print('Pearsons correlation: %.3f' % corr)


def draw_pearson (ColA, ColB, title, data_g, label_x):
  """
  the function draws the scatter plot
  :param ColA: X
  :param ColB: Y
  :param title: the title of the plot
  :param data_g: the data
  :param label_x: the labels of x
  :return: shows the plot
  """
  ax = sns.scatterplot(x=ColA, y=ColB, data=data_g)
  ax.set_title(title)
  ax.set_xlabel(label_x)

# Positive Corr with stock value
for trend, stock_value, name in zip(list_of_lists_positive, list_value_close, stocks_names_list):
    print(f"The Correlation between the positive tweets and the stock value of {name} is:")
    calculate_pearson(trend, stock_value[0])
    new_df = pd.DataFrame(trend, columns=['trend'])
    new_df['stock value'] = stock_value[0]
    draw_pearson('trend', 'stock value', 'Correlation of '+name+' stock value and positive trend',
                 new_df, 'number of positive')
    plt.show()
    print()

# Negative Corr with stock value
for trend, stock_value, name in zip(list_of_lists_neg, list_value_close, stocks_names_list):
    print(f"The Correlation between the negative tweets and the stock value of {name} is:")
    calculate_pearson(trend, stock_value[0])
    new_df = pd.DataFrame(trend, columns=['trend'])
    new_df['stock value'] = stock_value[0]
    draw_pearson('trend', 'stock value', 'Correlation of '+name+' stock value and negative trend',
                 new_df, 'number of positive')
    plt.show()
    print()


# Uncertainty Corr with stock value
for trend, stock_value, name in zip(list_of_lists_uncertain, list_value_close, stocks_names_list):
    print(f"The Correlation between the Uncertainty tweets and the stock value of {name} is:")
    new_df = pd.DataFrame(trend, columns=['trend'])
    new_df['stock value'] = stock_value[0]
    draw_pearson('trend', 'stock value', 'Correlation of '+name+' stock value and Uncertainty trend',
                 new_df, 'number of positive')
    calculate_pearson(trend, stock_value[0])
    print()