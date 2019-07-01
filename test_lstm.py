from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import clean
from clean import cleanup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict


path = 'datasets/valid.csv'
vector_dimension=300
data = pd.read_csv(path)
missing_rows = []
for i in range(len(data)):
	if data.loc[i, 'text'] != data.loc[i, 'text']:
        	missing_rows.append(i)
data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

for i in range(len(data)):
	data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

data = data.sample(frac=1).reset_index(drop=True)

v = data.loc[:,'text'].values
   
Len=len(v)



valid_size = int(Len)
vtr = v[:valid_size]
np.save('vtr_shuffled.npy',vtr)

top_words = 5000
epoch_num = 5
batch_size = 64

xtr = np.load('./xtr_shuffled.npy')
xte = np.load('./xte_shuffled.npy')
y_train = np.load('./ytr_shuffled.npy')
y_test = np.load('./yte_shuffled.npy')
vtr=np.load('./vtr_shuffled.npy')

cnt = Counter()
x_train = []
for x in xtr:
    x_train.append(x.split())
    for word in x_train[-1]:
        cnt[word] += 1

#Then we count the frequency of each word appeared in our training dataset to find 5000 most common words and give each one an unique integer ID
print("Storing most common words")
most_common = cnt.most_common(top_words + 1)
word_bank = {}
id_num = 1
for word, freq in most_common:
    word_bank[word] = id_num
    id_num += 1

#After that we replace each common word with its assigned ID and delete all uncommon words.
print( "Encode the sentences")
for news in x_train:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

y_train = list(y_train)
y_test = list(y_test)

print("Delete the short news")
i = 0
while i < len(x_train):
    if len(x_train[i]) > 10:
        i += 1
    else:
        del x_train[i]
        del y_train[i]

print("Generating test data")
x_test = []
for x in xte:
    x_test.append(x.split())

print("Encode the sentences")
for news in x_test:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]


print("Generating valid data")
v_test = []
for v in vtr:
    v_test.append(v.split())

print("Encode the sentences")
for news in v_test:
    i = 0
    while i < len(news):
        if news[i] in word_bank:
            news[i] = word_bank[news[i]]
            i += 1
        else:
            del news[i]

print("Truncate and pad input sequences")
max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
V_test=sequence.pad_sequences(v_test, maxlen=max_review_length)

print("Convert to numpy arrays")
y_train = np.array(y_train)
y_test = np.array(y_test)


print("Create the model")
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+2, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_num, batch_size=batch_size)
print("Final evaluation of the model")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy= %.2f%%" % (scores[1]*100))


y_pred = model.predict_classes(V_test)
print(y_pred)















