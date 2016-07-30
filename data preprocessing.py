# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 22:50:00 2016

@author: Cedric Oeldorf
"""
from __future__ import print_function

MAX_CHARACTERS_FROM_TEXT = 360000

SYMBOLS = '{}()[].,:;+-*/&|<>=~$'
ENCODING = 'UTF-8-SIG'
MAX_VOCABULARY_SIZE = 15000
HIDDEN_SIZE = 512
MAX_SEQ_LEN = 50 # sentences with more tokens than MAX_SEQ_LEN are filtered
BATCH_SIZE = 52

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import GRU, LSTM
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import nltk
import numpy as np
import sys
import codecs
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
# taken from: https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py and modified
def convert_sequences(sequences, max_nb_words=None,
                      maxlen=None, seed=113):

    start_char=1
    end_char=2
    oov_char=3

    if maxlen:
        new_sequences = []
        for s in sequences:
            if len(s) + 2 < maxlen:
                new_sequences.append(s)
        sequences = new_sequences
    if not sequences:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')

    print(len(sequences), "sentences remanied after filtering by MAX_SEQ_LEN")

    # Count the word frequencies
    word_freq = nltk.FreqDist([w for s in sequences for w in s])
    uniq_words = len(word_freq.items())
    print("Found %d unique tokens." % uniq_words)

    max_nb_words = min(max_nb_words, uniq_words+4)
    print("Using vocabulary size %d." % max_nb_words)

    # Get the most common words and build index_to_word and word_to_index vectors
    vocabulary = [u"<NULL>", u"<START>", u"<END>", u"<UNK>"] + [w for w, c in word_freq.most_common(max_nb_words-4)] # keep 4 slots for: padding=0, start=1, end=2 and oov=3
    word_indices = dict((w, i) for i, w in enumerate(vocabulary))
    indices_word = dict((i, w) for i, w in enumerate(vocabulary))

    # Convert words to indices and pad with start-end
    X = [[start_char] + [word_indices.get(w, oov_char) for w in s] + [end_char] for s in sequences]
    # create subsequences
    X = [x[:i] for x in X for i in range(1,len(x)+1)]

    np.random.seed(seed)
    np.random.shuffle(X)

    y = [x[-1] for x in X]
    X = [x[:-1] for x in X]

    X = sequence.pad_sequences(X, maxlen=maxlen)
    y = to_categorical(y, len(word_indices))
    
    return X, y, word_indices, indices_word


# 1. Import text and tokenize into sentences
path = "C:/Users/Cedric Oeldorf/Desktop/University/Research/Data/Gutenberg/kafka.txt"
with codecs.open(path, 'r', ENCODING) as f:
    if MAX_CHARACTERS_FROM_TEXT:
        text = f.read()[:MAX_CHARACTERS_FROM_TEXT].lower()
    else:
        text = f.read().lower()
sent_tokenize_list = sent_tokenize(text)
print('Number of characters:', len(text))
print('Number of sentences:', len(sent_tokenize_list))
print('First sentence:', sent_tokenize_list[0].encode(ENCODING))
del text

# 2. Clean sentences of surrounding symbols and tokenize into lists of tokens
tokens = [word_tokenize(sentence.strip(SYMBOLS)) for sentence in sent_tokenize_list]
print("First sentence tokens:", tokens[0])

# 3. Convert to inputs X and labels Y, and pad with start-end tokens, and replace rare words with UNK
print('Converting data...')
X, y, word_indices, indices_word = convert_sequences(tokens, max_nb_words=MAX_VOCABULARY_SIZE, maxlen=MAX_SEQ_LEN)
vocab_size = len(word_indices)
print('Got %d sequences' % len(X))
"""
print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, HIDDEN_SIZE, input_length=MAX_SEQ_LEN, mask_zero=True))
model.add(GRU(HIDDEN_SIZE/2,return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(HIDDEN_SIZE/2,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
"""
print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, HIDDEN_SIZE, input_length=MAX_SEQ_LEN, mask_zero=True))
model.add(LSTM(HIDDEN_SIZE))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.random.choice(len(a), p=a)
    #return np.argmax(np.random.multinomial(1, a, 1))

model.load_weights('C:/Users/Cedric Oeldorf/Desktop/University/Research/Code/MODELS/GRU_final_final_final.h5')

# train the model, output generated text after each iteration
from keras.callbacks import History
hist = History()
h = []
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    
    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1, callbacks=[hist], validation_split=0.1, show_accuracy=True)
    model.save_weights('C:/Users/Cedric Oeldorf/Desktop/University/Research/Code/MODELS/GRU_24June_bigmod.h5',overwrite=True)
    h.append(hist.history)
    for diversity in [1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = [word_indices["<NULL>"]] * (MAX_SEQ_LEN - 1) + [word_indices["<START>"]]

        for i in range(MAX_SEQ_LEN):
            
            preds = model.predict(np.array([generated]), verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            generated.append(next_index)
            generated = generated[-MAX_SEQ_LEN:]

            sys.stdout.write(next_word + " ")
            sys.stdout.flush()

            if next_word == "<END>":
                break
    print()

def save_loss(filename):
    list0 = []    
    list0 = [f['loss'] for f in h]  
    list2 = []
    list2 = [l[0] for l in list0]   
    list0 = []    
    list0 = [f['val_loss'] for f in h]  
    list3 = []
    list3 = [l[0] for l in list0]   
    import csv

    with open("C://Users//Cedric Oeldorf//Desktop//University//Research//RESULTS//GRU//300k//loss_model2.csv", 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list2)
    with open("C://Users//Cedric Oeldorf//Desktop//University//Research//RESULTS//GRU//300k//loss_val_model2.csv", 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list3)
save_loss("C://Users//Cedric Oeldorf//Desktop//University//sResearch//RESULTS//GRU//300k//loss_withval_512bigbatch.csv")


path_loss = "C://Users//Cedric Oeldorf//Desktop//University//Research//RESULTS//LSTM//300k//L1.csv"
path_val_loss = "C://Users//Cedric Oeldorf//Desktop//University//Research//RESULTS//LSTM//300k//L1_val.csv"
def load_loss(path_lstm_loss, path_gru_loss):
    import pandas as pd
    loss = pd.DataFrame()
    import csv
    with open(path_loss, 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        
    lstm = [item for sublist in your_list for item in sublist]        
    loss["loss"] = lstm
    with open(path_val_loss, 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    val = [item for sublist in your_list for item in sublist]
    
    loss["val"] = val

load_loss(path_loss,path_val_loss)


model.save_weights('C:/Users/Cedric Oeldorf/Desktop/University/Research/Code/MODELS/GRU_19June.h5')
from keras.utils.visualize_util import plot
plot(model, to_file='C:/Users/Cedric Oeldorf/Desktop/University/Research/Results/LSTM/kafka/LSTMmodel1.png')

#************************************************************************************

# PLOTTING

#************************************************************************************

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 14))

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
k = len(loss)
#range for whole, np arrange for decimal
plt.xticks(range(0, k, 2), fontsize=14)
plt.yticks(np.arange(0, 7, 0.2))

for y in np.arange(0, 7, 0.2):
    plt.plot(range(0,k), [y] * len(range(0,k)), '--',
             lw=0.5, color='black', alpha=0.3)
plt.tick_params(axis='both', which='both', bottom='on', top='off',
                labelbottom='on', left='off', right='off', labelleft='on')

plt.plot(loss['loss'],color='b',linewidth=2)
plt.axvline(x=12,color='g',ls='dashed',linewidth=2)

plt.plot(loss['val'],color='r',linewidth=2)
#plt.text(1, 0.8, 'LSTM', fontsize=14, color='b')
#plt.text(1, 0.8, 'GRU', fontsize=14, color='b')

import matplotlib.patches as mpatches

blue_patch = mpatches.Patch(color='b', label='Training Loss')
red_patch = mpatches.Patch(color='r', label='Validation Loss')
plt.legend(bbox_to_anchor=(1, 1),handles=[blue_patch,red_patch])
#plt.title("Minimization of cross entropy loss (GRU model 2)")
#plt.xlabel("Iteration")
#plt.ylabel("Cross entropy loss")
ax.set_title("Minimization of cross entropy loss (LSTM model 1)",fontsize=20)
ax.set_xlabel("Iteration",fontsize=20)
ax.set_ylabel("Cross entropy loss",fontsize=20)
#**************************************************************************

#Calculate perplexity

#**************************************************************************
loss = loss.convert_objects(convert_numeric=True)

loss["perp_lstm"] = 2**(loss["val"])
loss["perp_gru"] = 2**(loss["gru"])

    
