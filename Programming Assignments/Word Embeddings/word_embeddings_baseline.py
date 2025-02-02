# -*- coding: utf-8 -*-
"""Word_Embeddings_Baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15F4d1YeZHVol9zm5JISFobpQfiop02Lw
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

import numpy as np
from matplotlib import pyplot as plt

print(tf.__version__)

# Loading the dataset
(ds_train, ds_test), ds_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Prepping the data a little
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

vocab_size = 20000

tokenizer = Tokenizer(num_words = vocab_size, lower = True, oov_token = '<OOV>')
tokenizer.fit_on_texts(x_train)

maxlen = 50
padding = 'post'
truncating = 'post'

sequences = tokenizer.texts_to_sequences(x_train)
seq_train = pad_sequences(sequences, maxlen = maxlen, padding = padding, truncating = truncating)

test_sequences = tokenizer.texts_to_sequences(x_test)
seq_test = pad_sequences(test_sequences, maxlen = maxlen, padding = padding, truncating = truncating)

embedding_dim = 100

model = Sequential([
                    Embedding(input_dim = vocab_size+1, output_dim= embedding_dim, input_length = maxlen),
                    Bidirectional(LSTM(64)),
                    BatchNormalization(),
                    Dense(32, activation = 'relu'),
                    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(seq_train, y_train, batch_size=10000, epochs = 10, validation_data = (seq_test, y_test))

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

# Extracting the Embeddings
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]
print(weights.shape)

print(tokenizer.word_index)

# Saving the weights
import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word in tokenizer.word_index:
  num = tokenizer.word_index[word]
  vec = weights[num] # skip 0, it's padding.
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

# Downloading the weights

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')