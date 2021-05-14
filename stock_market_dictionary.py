import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import operator
import collections

lemma = WordNetLemmatizer()
nltk.download('stopwords')
stop = stopwords.words('english')

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
    """
    the function takes a full string of tweets and separtes it into tokens
    :param tweets: a string of tweets
    :return: tokens
    """
    tweets_tokens = [preprocess(tweet, True, False) for tweet in tweets]
    return tweets_tokens


def tokenize(s):
    """
    the function takes a specific token and checks if it fits to a regex
    :param s: a string
    :return: a fixed version of the token (if it fits to an emoji regex for example)
    """
    try:
        return tokens_re.findall(s)
    except Exception:
        return None


def normalize_word(text, removeSpecial=False):
    """
    the function takes a token and transform it to small letters and lemmatization
    :param text: the string of the tweet
    :param removeSpecial: tells the function not to clean special characters
    :return: a normalized tokens
    """
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
    """
    the function cleans tokens from some characters or line up the format
    :param tweet: the string of the tweet
    :return: a clean and formatted token
    """
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
    """
    the function takes all of the tweets and preprocess them so we can do additional statistic
    :param s: the string of the tweet
    :param lowercase: boolean of lowercase conversation
    :param toStringList: boolean if to convert it back to a string
    :return: string or separated tokens
    """
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


# ------------------------------------------------------------------------------------------------- #
def prepare_dictionary(list_of_words):
    """
    gets a list of words and creates a set after normalizing the list
    :param list_of_words: the list of words
    :return: a set
    """
    stop_free = [i.strip() for i in list_of_words if i not in stop and i and i == i]
    lower_case = [word.lower() for word in stop_free if word and word == word]
    normalized = [lemma.lemmatize(word) for word in lower_case if word]
    new_set = set(normalized)
    return new_set


dictionary_of_stocks = pd.read_csv("stock_market_dictionary.csv")
set_negative = prepare_dictionary(list(dictionary_of_stocks['Negative']))
set_positive = prepare_dictionary(list(dictionary_of_stocks['Positive']))
set_uncertainty = prepare_dictionary(list(dictionary_of_stocks['Uncertainty']))

print(set_negative)
print("--------------------")
print(set_positive)
print("--------------------")
print(set_uncertainty)
print("-------------------------------------------------")


def split_a_tweet(row):
    """
    splits a tweet in a df
    :param row: the tweet
    :return: list of tokens
    """
    return row.split()


netflix = pd.read_csv("nflx.csv")


def count_trend(row):
    """
    count how many times a certain word from the stock market dictionary exists
    :param row: the row of a df
    :return: the most common behaviour
    """
    count_positive = 0
    count_negative = 0
    count_uncertainty = 0
    if row:
        if len(row) > 0:
            for item in row:
                if item in set_positive:
                    count_positive += 1
                elif item in set_negative:
                    count_negative += 1
                elif item in set_uncertainty:
                    count_uncertainty += 1
    if count_negative + count_negative + count_uncertainty == 0:
        return -1
    stats = {"positive": count_positive, "negative": count_negative, "uncertainty": count_uncertainty}
    result = max(stats.items(), key=operator.itemgetter(1))[0]
    return result

netflix['tokens'] = netflix['tweet'].apply(split_a_tweet)
netflix['tokens'] = netflix['tokens'].apply(normalize_word)
netflix['trend'] = netflix['tokens'].apply(count_trend)
c = collections.Counter(list(netflix['trend']))
print(c)
