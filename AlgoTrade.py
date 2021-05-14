import pandas as pd
import datetime
from nltk.corpus import stopwords
import string
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter

# define the variables of the program
lemma = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
ps = PorterStemmer()


facebook = pd.read_csv("fb.csv", lineterminator='\n')
amazon = pd.read_csv("amzn.csv", lineterminator='\n')
apple = pd.read_csv("appl.csv", lineterminator='\n')
google = pd.read_csv("goog.csv", lineterminator='\n')
tesla = pd.read_csv("tsla.csv", lineterminator='\n')
netflix = pd.read_csv("nflx.csv", lineterminator='\n')
list_stocks = [facebook, amazon, apple, google, tesla, netflix]
list_names_stocks = ["facebook", "amazon", "apple", "google", "tesla", "netflix"]


def change_format_of_dates_line(row):
    """
    the function converts the format of the days of the dataframe
    :param row: the chosen row
    :return: a formatted string
    """
    try:
        temp = datetime.datetime.strptime(row, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        return temp
    except Exception:
        return "2021-01-01"


def change_format_of_dates_slash(row):
    """
    the function converts the format of the days of the dataframe
    :param row: the chosen row
    :return: a formatted string
    """
    try:
        temp = datetime.datetime.strptime(row, '%m/%d/%Y %H:%M').strftime('%Y-%m-%d')
        return temp
    except Exception:
        return "2021-01-01"


amazon['date'] = amazon['date'].apply(change_format_of_dates_slash)
netflix['date'] = netflix['date'].apply(change_format_of_dates_slash)
tesla['date'] = tesla['date'].apply(change_format_of_dates_slash)
facebook['date'] = facebook['date'].apply(change_format_of_dates_slash)
apple['date'] = apple['date'].apply(change_format_of_dates_slash)
google['date'] = google['date'].apply(change_format_of_dates_slash)

# convert the string column to a datetime column
for df, name in zip(list_stocks, list_names_stocks):
    try:
        print(f"Dataframe Name: {name}")
        print(f"Number of records before the cleaning process: {df.shape}")
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df = df[(df['date'] >= '2021-05-04') & (df['date'] <= '2021-05-11')]
        print(f"Number of records after filtering dates: {df.shape}")
        df = df.drop_duplicates(subset='tweet', keep="last")
        print(f"Number of records after removing duplicates: {df.shape}")
        list_stocks.append(df)
    except Exception:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M')
        df = df[(df['date'] >= '5/4/2021') & (df['date'] <= '5/11/2021')]
        print(f"Number of records after filtering dates: {df.shape}")
        df = df.drop_duplicates(subset='tweet', keep="last")
        print(f"Number of records after removing duplicates: {df.shape}")
        list_stocks.append(df)
    print()


list_stocks = list_stocks[0:6]

for item in list_stocks:
    print(item.shape)
# Code from the lecture in order to clean the tweets

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'the', u'\u2019', u'\u2026',
                                                   'The', u'de', u'\xe9',
                                                   'ï', '¿', '\u200d', '\u200b']

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def transform_tweets_to_tokens(tweets):
    tweets_tokens = [preprocess(tweet, True, False) for tweet in tweets]
    return tweets_tokens


def tokenize(s):
    # return tokens_re.findall(s.encode('ascii','replace'))
    try:
        return tokens_re.findall(s)
    except Exception:
        return None


def normalize_word(text, removeSpecial=False):
    exclude = set(string.punctuation)
    stop_free = []
    if removeSpecial:
      no_special = [CleanTweet(t) for t in text]
      stop_free = [i.strip() for i in text if i not in stop and i]
    else:
      stop_free = [i.strip() for i in text if i not in stop and i]
    normalized = [lemma.lemmatize(word) for word in stop_free]
    return normalized

def CleanTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\.', ' . ', tweet)
    tweet = re.sub(r'\!', ' !', tweet)
    tweet = re.sub(r'\?', ' ?', tweet)
    tweet = re.sub(r'\,', ' ,', tweet)
    tweet = re.sub(r':', ' : ', tweet)
    tweet = re.sub(r'#', ' # ', tweet)
    tweet = re.sub(r'@', ' @ ', tweet)
    tweet = re.sub(r' amp ', ' and ', tweet)
    tweet = re.sub(r' . . . ', ' ', tweet)
    tweet = re.sub(r' .  .  . ', ' ', tweet)
    tweet = re.sub(r' ! ! ', ' ! ', tweet)
    tweet = re.sub(r'&amp', 'and', tweet)
    tweet = re.sub('[^A-Za-z0-9]+', '', tweet)
    return tweet


def preprocess(s, lowercase=False, toStringList=False):
    tokens = tokenize(s)
    if not tokens:
        return ""
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        if toStringList:
          tokens = normalize_word(tokens, True)
        else:
          tokens = normalize_word(tokens)
    if toStringList:
        tokens = [x for x in tokens if x]
        tokens = ' '.join(tokens)
    return tokens


def find_all_hashtags(row):
    try:
        hashtags = [i for i in row.split() if i.startswith("#")]
        return hashtags
    except Exception:
        return []


def find_all_tags(row):
    try:
        tags = [i for i in row.split() if i.startswith("@")]
        return tags
    except Exception:
        return []


def create_df_from_list_tuple(list_tup):
    list1, list2 = zip(*list_tup)
    df = pd.DataFrame(list1, columns=['word'])
    df['frequency'] = list2
    return df


def create_df_from_list_tuple_two_words(list_tup):
    list1, list2 = zip(*list_tup)
    df = pd.DataFrame(list(list1), columns=['first word', 'second word'])
    df['frequency'] = list2
    return df


def validate_top_unigrams(l_unigrams):
  counter = 0
  verified_list = []
  for p in l_unigrams:
    if p[0] and p[0] != '️' and p[0] != 'quot':
      verified_list.append(p)
      counter = counter + 1
    if counter == 10:
      break
  return verified_list


def validate_top_bigrams(l_bigrams):
  counter = 0
  verified_list = []
  for p in l_bigrams:
    exp = p[0]
    if exp[0] != '' and exp[0] != 'lt' and exp[1] != '️':
      verified_list.append(p)
      counter = counter + 1
    if counter == 10:
      break
  return verified_list



def show_most_common_terms(df):

    # transform the strings to tokens
    tokens = transform_tweets_to_tokens(df['tweet'].values)
    flat_tokens = [item for sublist in tokens for item in sublist]

    # most 10 common terms
    tokens_unigrams = nltk.FreqDist(flat_tokens)
    tokens_bigrams = nltk.FreqDist(nltk.bigrams(flat_tokens))
    top_10_unigrams = validate_top_unigrams(tokens_unigrams.most_common(50))
    top_10_bigrams = validate_top_bigrams(tokens_bigrams.most_common(50))
    df_unigrams = create_df_from_list_tuple(top_10_unigrams)
    print(df_unigrams)
    df_bigrams = create_df_from_list_tuple_two_words(top_10_bigrams)
    print(df_bigrams)


for df, name in zip(list_stocks, list_names_stocks):
    print(f"The most common terms of {name} stock are:")
    print()
    show_most_common_terms(df)
    print()


def show_most_popular_tags_hashtags(df):
  all_tweets = df.loc[df['tweet'] != None]
  all_tweets['all_hashtags'] = all_tweets['tweet'].apply(find_all_hashtags)
  all_tweets['all_tags'] = all_tweets['tweet'].apply(find_all_tags)

  tweet_hashtags = list(all_tweets['all_hashtags'])
  tweet_tags = list(all_tweets['all_tags'])

  # create one flat list
  flat_hashtags = [item for sublist in tweet_hashtags for item in sublist]
  flat_tags = [item for sublist in tweet_tags for item in sublist]

  count_hashtags = Counter(flat_hashtags)
  count_tag = Counter(flat_tags)

  df1 = pd.DataFrame(count_hashtags.items(), columns=['Hashtag', 'Count']).sort_values(by=['Count'], ascending=False)
  df2 = pd.DataFrame(count_tag.items(), columns=['Tag', 'Count']).sort_values(by=['Count'], ascending=False)

  df1 = df1[df1.Hashtag.apply(lambda x: len(str(x))>1)]
  df2 = df2[df2.Tag.apply(lambda x: len(str(x))>1)]

  blankIndex = [''] * len(df1)
  df1.index = blankIndex
  blankIndex = [''] * len(df2)
  df2.index = blankIndex

  print()
  print("Most popular hashtags:")
  print()
  print(df1.head(10))
  print()
  print("Most popular tags:")
  print(df2.head(10))


for df, name in zip(list_stocks, list_names_stocks):
    print(f"The most common Hashtags and Tags of {name} stock are:")
    print()
    show_most_popular_tags_hashtags(df)
    print()


def sum_tweets_per_day(list_stocks, list_names_stocks):
    list_amount_tweets = []
    for df, name in zip(list_stocks, list_names_stocks):
        print(f"Count per day in {name}:")
        sum_tweets_per_day = df.groupby('date')['tweet'].count().reset_index(name="count")
        list_amount_tweets.append(sum_tweets_per_day)
        print(sum_tweets_per_day)
        print()
    return list_amount_tweets


sum_tweets_per_day(list_stocks, list_names_stocks)