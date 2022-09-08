import pandas as pd 

import pickle
import re
import os

import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
#nltk.download('omw-1.4')

# for preprocessing and cleaning
import preprocessor as p 

# used for stop words removal
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

#------------------------------- Cleaning the tweets
def preproc(x):
    # removing the html parsers
    def clean_html_inserts(text, split=False):
        if split:
            sep = os.linesep
        else:
            sep = ' '
        text = re.sub(r"<(\w|\W)+?>", sep, text)
        return text

    # removing special characters
    def clean_symbols(text, specifics=[]):
        chars_to_clean = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.','/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_','`', '{', '|', '}', '~', 'Â»', 'Â«', 'â€œ', 'â€']
        chars_to_split = ["'"]
        chars_to_clean.extend(specifics)
        punct_pattern = re.compile("[" + re.escape("".join(chars_to_clean)) + "]")
        text = re.sub(punct_pattern, "", text)
        split_pattern = re.compile("[" + re.escape("".join(chars_to_split)) + "]")
        text = re.sub(split_pattern, " ", text) 
        text = shorten_whitespaces(text)
        return text

    # replace multiple white spaces
    def shorten_whitespaces(text):
        text = re.sub('\s{2,}', " ", text)
        return text

    # removing @usernames
    def remove_names(text):
        text=" ".join(filter(lambda x:x[0]!='@', text.split()))
        return text

    # lemmatizing sentances
    def lemmatize_text(text):
        return [(WordNetLemmatizer().lemmatize(w)) for w in \
                                        TweetTokenizer().tokenize((text))]

    # lemmatizing and processing the tweets
    x=" ".join(lemmatize_text(remove_stopwords(remove_names(clean_symbols(clean_html_inserts(p.clean(x.strip().lower())))))))
    
    x=pd.Series(x)
    #print(x)

    # load the Vectorizer.
    loaded_count_vector = pickle.load(open('vectorizer.pickle', 'rb'))
    # transform the data and return the matrix.
    x_predict = loaded_count_vector.transform(x)

    return x_predict
