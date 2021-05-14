import pandas as pd
import matplotlib.pyplot as plt

'''
def cut_label_google_trends(row):
    """
    converts the date from google trends csv to new format
    :param row: the row in the df
    :return: a new formatted date
    """
    values = row.split("/")
    month = values[0]
    day = values[1]
    return str(day)+"-"+str(month)+"-21"

google_trend = pd.read_csv("google_trends.csv")
google_trend['date'] = google_trend['date'].apply(cut_label_google_trends)
google_trend = google_trend.set_index(['date'])
lines = google_trend.plot.line()
plt.title("Google Trends per day")
plt.ylabel("Ratio amount of searches")
plt.legend(bbox_to_anchor=(0.9, 0.6))
plt.show()
'''
