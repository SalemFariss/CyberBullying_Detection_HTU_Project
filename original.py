import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from time import time

from sklearn.linear_model import SGDClassifier
import pickle
import re
import os
# important libraries for preprocessing using NLTK speacially for lematization
from nltk.stem import WordNetLemmatizer
#nltk.download
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
#nltk.download('omw-1.4')

# for preprocessing and cleaning
import preprocessor as p 

# used for stop words removal
from gensim.parsing.preprocessing import remove_stopwords

#--------------------- Cleaning the tweets

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
lemmatizer = WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in \
                                     w_tokenizer.tokenize((text))]

#------------------------- Loading and Analyzing the data 
df_tweets = pd.read_csv('tweets.csv')
df_public = pd.read_csv('public_data_labeled.csv')

# removing duplicates
df_tweets.drop_duplicates(inplace = True)
df_tweets.drop('id', axis = 'columns', inplace = True)
df_public.drop_duplicates(inplace = True)

# Concatenate dfs
df = pd.concat([df_tweets, df_public])

col= df.columns
df=df.values.tolist() # df to list

# lemmatizing and processing the tweets (Cleaning the tweets)
from tqdm import tqdm
for i in tqdm(range(len(df))):
    df[i][1]=" ".join(lemmatize_text(remove_stopwords(remove_names(clean_symbols(clean_html_inserts(p.clean(df[i][1].strip().lower())))))))


df=pd.DataFrame(df,columns=col)

print()
# Distribution of Tweets in the Dataset
print('Total Tweets:', df.shape[0])
sorted_counts = df['label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
print(sorted_counts)


df['label'] = df.label.map({'Offensive': 1, 'Non-offensive': 0})

print('---------------------------------------')

#----------------------- Implementing -
# Separate training and testing data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'],test_size=0.25, random_state=42)
print('The total set: {}'.format(df.shape[0]))
print('The training set: {}'.format(X_train.shape[0]))
print('The test set: {}'.format(X_test.shape[0]))

print('---------------------------------------')


# Vectorize the data
# Instantiate CountVectorizer method.
count_vector = CountVectorizer(stop_words = 'english', lowercase = True)
# Fit the training data and return the matrix.
X_train = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. (not fitting)
X_test = count_vector.transform(X_test)


#Learning the model:

# The model name:
model =SGDClassifier()


test_results = {}
test_res=[]
train_results = {}
train_res=[]

print("Training {}".format(model.__class__.__name__))
# Fit the model:
start = time()
model = model.fit(X_train, y_train)
end = time() 

# The training time
training_time = end - start
print('Training Time: {}'.format(training_time))

start = time()
predictions_test = model.predict(X_test)
predictions_train = model.predict(X_train)
end = time()

# The prediction time
prediction_time = end - start
print('Prediction Time: {}'.format(prediction_time))
print()
# Compute the Accuracy
test_results['Accuracy Test'] = accuracy_score(y_test, predictions_test)
train_results['Accuracy Train'] = accuracy_score(y_train, predictions_train)

# Compute the F1 Score
test_results['F1 Score Test'] = f1_score(y_test, predictions_test)
train_results['F1 Score Train'] = f1_score(y_train, predictions_train)

# Compute the Precision
test_results['Precision Test'] = precision_score(y_test, predictions_test)
train_results['Precision Train'] = precision_score(y_train, predictions_train)

# Compute the Recall
test_results['Recall Test'] = recall_score(y_test, predictions_test)
train_results['Recall Train'] = recall_score(y_train, predictions_train)

test_res.append(test_results.copy())
train_res.append(train_results.copy())

res1 = pd.DataFrame(test_res)
res2 = pd.DataFrame(train_res)

print(res1)
print(res2)
print()
print("Training {} finished in {} sec".format(model.__class__.__name__, training_time))

from sklearn.metrics import classification_report, confusion_matrix
print()

print()
print(confusion_matrix(y_test,predictions_test))
print(classification_report(y_test,predictions_test))



# Save the vectorizer
vec_file = 'vectorizer.pickle'
pickle.dump(count_vector, open(vec_file, 'wb'))

# Save the model
mod_file = 'classification.model'
pickle.dump(model, open(mod_file, 'wb'))