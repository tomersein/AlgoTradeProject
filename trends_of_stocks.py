import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt

facebook = pd.read_csv("./filtered_stocks/facebook.csv", lineterminator='\n')
amazon = pd.read_csv("./filtered_stocks/amazon.csv", lineterminator='\n')
apple = pd.read_csv("./filtered_stocks/apple.csv", lineterminator='\n')
google = pd.read_csv("./filtered_stocks/google.csv", lineterminator='\n')
tesla = pd.read_csv("./filtered_stocks/tesla.csv", lineterminator='\n')
netflix = pd.read_csv("./filtered_stocks/netflix.csv", lineterminator='\n')
list_stocks_filtered = [facebook, amazon, apple, google, tesla, netflix]
names = ["facebook", "amazon", "apple", "google", "tesla", "netflix"]

def remove_r(row):
    row = row.replace('\r', "")
    return row

list_sum_trends = []
for item, name in zip(list_stocks_filtered, names):
    item['trend\r'] = item['trend\r'].apply(remove_r)
    sum_tweets_per_day = item.groupby(['date','trend\r'])['trend\r'].count().reset_index(name="count_trend")
    cols = ['date', 'trend', 'count_trend']
    sum_tweets_per_day.columns = cols
    print(f"The trends df of {name}:")
    print(sum_tweets_per_day)
    print()
    list_sum_trends.append(sum_tweets_per_day)


def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height
    :param ax: the plot object
    :param rects: the labels
    :return: formatted labels
    """
    for rect in rects:
        height = round(rect.get_height(),2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def show_plots_of_compare_trends_stock(results, title):
    """
    the function creates the plot and shows
    :param results: a list of tuple
    :param title: the title of the plot
    :return: the plot
    """
    results = [[x[i] for x in results] for i in range(4)]
    # [facebook, amazon, apple, google, tesla, netflix]
    days, tweets_pos, tweets_neg, tweet_un = results
    tweets_pos_g = np.array(tweets_pos) / np.max(tweets_pos)
    tweets_neg_g = np.array(tweets_neg) / np.max(tweets_neg)
    tweet_un_g = np.array(tweet_un) / np.max(tweet_un)


    labels = []
    for b, c, d, in zip(tweets_pos_g, tweets_neg_g, tweet_un_g):
      labels.append(round(b,2))
      labels.append(round(c,2))
      labels.append(round(d,2))

    a = np.arange(8)
    w = 0.1
    fig, ax = plt.subplots(figsize=(30, 7), edgecolor='k')
    ax.set_xticklabels(days)
    p1 = ax.bar(a+w, tweets_pos_g, w, color='cornflowerblue')
    p2 = ax.bar(a-w, tweets_neg_g, w, color='peachpuff')
    p3 = ax.bar(a, tweet_un_g, w, color='lightpink')
    ax.set_xticks(a)
    ax.set_title(title)
    # Evaluation of the models
    ax.legend((p1[0], p2[0], p3[0]),
              ('positive', 'negative', 'uncertainty'))
    plt.xlabel('Days')
    plt.ylabel('Ratio number of tweets')
    autolabel(ax, p1)
    autolabel(ax, p2)
    autolabel(ax, p3)
    plt.show()

for item in list_sum_trends:
    pos = item[(item['trend'] == 'positive')]
    neg = item[(item['trend'] == 'negative')]
    uncertainty = item[(item['trend'] == 'uncertainty')]
    list_pos = list(pos['count_trend'])
    list_neg = list(neg['count_trend'])
    list_uncertainty = list(uncertainty['count_trend'])
    print(list_pos)
    print(list_neg)
    print(list_uncertainty)
    all_data_to_graph = []
    new_labels = ['04-05-21', '05-05-21', '06-05-21', '07-05-21',
                  '08-05-21', '09-05-21', '10-05-21', '11-05-21']
    for a, b, c, d in zip(list(new_labels),
                                   list(list_pos),
                                   list(list_neg),
                                   list(list_uncertainty)):
        all_data_to_graph.append((a, b, c, d))
    print(all_data_to_graph)
    show_plots_of_compare_trends_stock(all_data_to_graph, 'Visual Compare between the days - trend analysis')