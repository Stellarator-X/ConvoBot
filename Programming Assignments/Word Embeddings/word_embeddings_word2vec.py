from keras.models import Model
from keras.layers import Input, Dense, Reshape, Embedding, BatchNormalization, LSTM, Bidirectional, Dropout
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

print(tf.__version__)

def download(filename, url, expected_bytes):
    # Downloading the dataset
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename)
    return filename

def read_data(filename):
  
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    # Creating a dataset from input corpus
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = download('text8.zip', url, 31344016)
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary 
    return data, count, dictionary, reverse_dictionary

vocab_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:7])

window_size = 3
vector_dim = 300
epochs = 200000

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# Building the Model 

input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)
similarity = keras.layers.Dot(axes=1, normalize=True)([target, context]) # Cos similarity
dot_product = keras.layers.Dot(axes=1)([target, context])
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.summary()

validation_model = Model(input=[input_target, input_context], output=similarity)


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()

# Extracting the Embeddings
embedding_layer = model.layers[2]
weights = embedding_layer.get_weights()[0]
print(weights.shape)

word_index = dictionary
# Saving the weights
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word in word_index:
  vec = weights[ int(word_index[word])]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

# Training on the imdb dataset

import tensorflow_datasets as tfds

# Loading the dataset
(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

x_train, y_train, x_test, y_test = [], [], [], []

for x in ds_train:
  text = str(x[0].numpy()).split("b", 1)[1][1:-1]
  label = x[1].numpy()
  x_train.append(text)
  y_train.append(label)

for x in ds_test:
  text = str(x[0].numpy()).split("b", 1)[1][1:-1]
  label = x[1].numpy()
  x_test.append(text)
  y_test.append(label)

x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

print(x_train.shape, y_train.shape)

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(x_train)

def text2seq(texts, dictionary):
  result = []
  for text in texts:
    text = str(text)
    r = [dictionary[word] for word in text.split() if word in dictionary]
    result.append(r)
  return result

maxlen = 100
padding = 'post'
truncating = 'post'

sequences = tokenizer.texts_to_sequences(x_train)
sequences = tokenizer.sequences_to_texts(sequences)
sequences = text2seq(sequences, dictionary)

seq_train = pad_sequences(sequences, maxlen = maxlen, padding = padding, truncating = truncating)

test_sequences = tokenizer.texts_to_sequences(x_test)
test_sequences = tokenizer.sequences_to_texts(sequences)
test_sequences = text2seq(sequences, dictionary)
seq_test = pad_sequences(test_sequences, maxlen = maxlen, padding = padding, truncating = truncating)

from keras.models import Sequential
from keras import regularizers

model = Sequential([
                    Embedding(input_dim = vocab_size, output_dim= 300, input_length = maxlen, weights = [weights], trainable = False),
                    LSTM(64, kernel_regularizer=regularizers.l2(0.001), return_sequences=True),
                    Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(0.001), return_sequences=True)),
                    Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(0.001))), 
                    Dense(64, activation = 'relu', kernel_regularizer='l2'),
                    Dropout(0.6),
                    Dense(32, activation = 'relu', kernel_regularizer='l2'),
                    Dropout(0.6),
                    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(seq_train, y_train, epochs = 10, validation_data = (seq_test, y_test), verbose = 1)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) 

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()

plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.show()