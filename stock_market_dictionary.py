import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
nltk.download('stopwords')
stop = stopwords.words('english')


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


df = pd.read_csv("stock_market_dictionary.csv")
set_negative = prepare_dictionary(list(df['Negative']))
set_positive = prepare_dictionary(list(df['Positive']))
set_uncertainity = prepare_dictionary(list(df['Uncertainty']))

print(set_negative)
print("--------------------")
print(set_positive)
print("--------------------")
print(set_uncertainity)
