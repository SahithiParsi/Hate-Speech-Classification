
# coding: utf-8

# # Hate Speech Classification - Cute 4

# In[43]:

# libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
np.random.seed(32)

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.layers.embeddings import Embedding

## Plotly

# import plotly.offline as py
# import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)

# Others
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
import os
import re
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


get_ipython().magic('matplotlib inline')


# ## The dataset

# In[44]:

df=pd.read_csv('C:/Users/sahithi/Desktop/AI_Cute/train.csv')
test_df=pd.read_csv('C:/Users/sahithi/Desktop/AI_Cute/test.csv')


# In[45]:

df.head()


# In[46]:

test_df.head()


# ## Text Cleaning

# In[47]:

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


# In[48]:

df['text'] = df['text'].map(lambda x: clean_text(x))
print(df.head())
test_df['text'] = test_df['text'].map(lambda x: clean_text(x)) ;
print(test_df.head())


# ## Target Creation

# In[49]:

# Function to encoding target
def createLable(row):
    if row['hate_speech'] == 0 and row['obscene'] ==  0 and row['insulting'] == 0:
        x = 0
    elif row['hate_speech'] == 0 and row['obscene'] ==  0 and row['insulting'] == 1:
        x = 1
    elif row['hate_speech'] == 0 and row['obscene'] ==  1 and row['insulting'] == 0:
        x = 2
    elif row['hate_speech'] == 0 and row['obscene'] ==  1 and row['insulting'] == 1:
        x = 3
    elif row['hate_speech'] == 1 and row['obscene'] ==  0 and row['insulting'] == 0:
        x = 4
    elif row['hate_speech'] == 1 and row['obscene'] ==  1 and row['insulting'] == 0:
        x = 5
    elif row['hate_speech'] == 1 and row['obscene'] ==  0 and row['insulting'] == 1:
        x = 6
    else:
        x = 7
    return x


# In[50]:

df['target'] = df.apply(createLable, axis = 1)


# In[51]:

df.head()


# We will only consider the  text of the reviews and the target.

# In[52]:

df['target'].value_counts()


# - 0 means comment is not  'hate_speech', 'obscene' and 'insulting'
# 
# - 1 means comment is only 'insulting'
# 
# - 2 means comment is only 'obscence'
# 
# - 3 means comment is 'obscene' and 'insulting'
# 
# - 4 means comment is 'hate_speech'
# 
# - 5 means comment is 'hate_speech' and 'obscene'
# 
# - 6 means comment is 'hate_speech' and 'insulting'
# 
# - 7 means comment is 'hate_speech', 'obscene' and 'insulting'

# In[53]:

plt.hist(df['target'],color='green')
plt.xlabel('Comments Type')
plt.ylabel('comments count')
plt.title('Histogram for Comments')
# plt.grid(True)
plt.annotate("0 - comment is not  'hate_speech', 'obscene' and 'insulting' \n 1 - comment is only 'insulting' \n 2 - comment is only 'obscence' \n 3 - comment is 'obscene' and 'insulting'\n4 - comment is 'hate_speech'\n5 - comment is 'hate_speech' and 'obscene'\n6 - comment is 'hate_speech' and 'insulting'\n7 - comment is 'hate_speech', 'obscene' and 'insulting'",
             xy=(0.9, 5500))


# In[54]:

df.groupby('target')['hate_speech'].count()/df['target'].count()


# Train data has 49.8% has good comments and 50.2% has bad comments

# In[55]:

df.dtypes


# Convert taget to categorical

# In[56]:

df['target'] = df['target'].astype('category')


# In[57]:

df.dtypes


# ### Train test Split

# In[58]:

train_text, test_text, train_y, test_y = train_test_split(df['text'],df['target'],test_size = 0.2)


# In[59]:

print('train: ', train_text.shape)
print('test: ', test_text.shape)


# The following cells uses Keras to preprocess text:
# - using a tokenizer. You may use different tokenizers (from scikit-learn, NLTK, custom Python function etc.). This converts the texts into sequences of indices representing the `20000` most frequent words
# - sequences have different lengths, so we pad them (add 0s at the end until the sequence is of length `1000`)
# - we convert the output classes as 1-hot encodings

# In[60]:

train_text.head()


# In[61]:

test_text.head()


# In[62]:

test_df.head()


# In[63]:

MAX_NB_WORDS = 20000

# get the raw text data
texts_train = train_text.astype(str)
texts_test = test_text.astype(str)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
sequences_testdf = tokenizer.texts_to_sequences(test_df['text'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# The tokenizer object stores a mapping (vocabulary) from word strings to token ids that can be inverted to reconstruct the original message (without formatting):

# In[64]:

type(tokenizer.word_index), len(tokenizer.word_index)


# In[65]:

index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())


# In[66]:

" ".join([index_to_word[i] for i in sequences[0]])


# In[67]:

index_to_word[46]


# 
# Let's have a closer look at the tokenized sequences:

# In[68]:

seq_lens = [len(s) for s in sequences]
print("average length: %0.1f" % np.mean(seq_lens))
print("max length: %d" % max(seq_lens))


# In[69]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(seq_lens, bins=50);


# 
# Let's zoom on the distribution of regular sized posts. The vast majority of the posts have less than 200 symbols:

# In[70]:

plt.hist([l for l in seq_lens if l < 200], bins=50);


# In[71]:

MAX_SEQUENCE_LENGTH = 150

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
x_test_df = pad_sequences(sequences_testdf, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)
print('Shape of data test df tensor:', x_test_df.shape)


# In[72]:

y_train = train_y
y_test = test_y

y_train = to_categorical(np.asarray(y_train))
print('Shape of label tensor:', y_train.shape)


# ## A complex model : LSTM

# In[73]:

from keras.models import Sequential
model = Sequential()
model.add(Embedding(20000, 100, input_length=150))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(8, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

# model.fit(x_train, y_train, validation_split=0.1,
#           nb_epoch=2, batch_size=128)
model.fit(x_train, y_train, validation_split=0.4, epochs=8)


# In[281]:

pred = model.predict_classes(x_test)


# In[300]:

confusion_matrix(y_test, pred)


# In[288]:

final = model.predict_classes(x_test_df)


# In[301]:

result = pd.DataFrame({'id': test_df['id'], 'target': final})


# In[302]:

result['hate_speech'] = result.apply(lambda row: 1 if row['target'] in [4,5,6,7] else 0, axis=1 )
result['obscene'] = result.apply(lambda row: 1 if row['target'] in [2,3,5,7] else 0, axis=1 )
result['insulting'] = result.apply(lambda row: 1 if row['target'] in [1,3,6,7] else 0, axis=1 )


# In[303]:

result.head()


# In[297]:

result = result.drop('target', axis = 1)


# In[298]:

result.head()


# In[299]:

result.to_csv("result1.csv", index=False)


# # Baseline Model

# In[11]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[12]:

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[18]:

# load dataset
dataframe = pandas.read_csv("Train.csv")
print(dataframe.head())
dataframe['text'] = dataframe['text'].map(lambda x: clean_text(x))
print(dataframe.head())
dataframe['target'] = dataframe.apply(createLable, axis = 1)
print(dataframe.head())


# In[19]:

MAX_NB_WORDS = 20000

# get the raw text data
texts_train = dataframe['text'].astype(str)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)


# In[23]:

MAX_SEQUENCE_LENGTH = 150

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print(x_train)


# In[24]:

X = x_train
Y = dataframe['target']


# In[28]:

Y[:5]


# In[29]:

# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(Y)


# In[30]:

dummy_y[:5]


# In[40]:

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=150, activation='relu'))
	model.add(Dense(8, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[41]:

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)


# In[42]:

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)


# In[ ]:

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
print(predictions)
print(encoder.inverse_transform(predictions))

