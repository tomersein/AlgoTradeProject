import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os





api= '0KADFWF9I271S9NZ'
def load_data(item):
    data_source = 'alphavantage' # alphavantage or kaggle

    if data_source == 'alphavantage':
        # ====================== Loading Data from Alpha Vantage ==================================

        api_key = 'IP48OK60PS9NF24L'

        # American Airlines stock market prices
        ticker = item

        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

        # Save data to this file
        file_to_save = 'stock_market_data-%s.csv'%ticker

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
            print('Data saved to : %s'%file_to_save)
            df.to_csv(file_to_save)
            return df

        # If the data is already there, just load it from the CSV
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save)
            return df

    else:

        # ====================== Loading Data from Kaggle ==================================
        # You will be using HP's data. Feel free to experiment with other data.
        # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
        df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
        print('Loaded data from the Kaggle repository')
        return df
def visualization_of_stock(df,name):
    df = df.sort_values('Date')
    # df = df.iloc[::-1]
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), df['Close'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    plt.xticks(df.index[::-1], df["Date"].values)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.title(name)
    # plt.show()
    plt.savefig(name)

if __name__ == "__main__":
    tickers =['AMZN','AAPL','FB','GOOG','NFLX','TSLA']
    for item in tickers:
        df=load_data(item)
        features = ['Date','Low','High','Close','Open']
        df=df[features]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= "2021-05-04") & (df['Date'] <= "2021-05-11")]
        file_to_save = 'stock_market_data-%s.csv' % item
        df.to_csv(file_to_save)
        visualization_of_stock(df,item)
