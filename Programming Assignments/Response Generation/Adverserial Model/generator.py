"""
Generator for response generation with an Adverserial Framework

@Author - Abhay Dayal Mathur
Contains the definitions for the generator with simple attention
Future work - try with multi-head attention/scaled dot product attention
"""
TOKENS_FILE = None # Str path to tokens file


# Getting the weights for the embedding layer
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
with open(TOKENS_FILE) as file:
    js_string = file.read()
tokenizer = tokenizer_from_json(js_string)
word_index = tokenizer.word_index
index_word = {word_index[word]:word for word in word_index}

import os
if "bin/glove.42B.300d.txt" not in os.popen("ls").read():
    print("Download Embeddings first")
    raise "s"

embedding_dim = 300
with open('bin/glove.42B.300d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;
embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

from tensorflow.keras.layers import *
import tensoflow as tf 

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], trainable=False)
        self.gru = GRU(self.encoder_units, return_sequences = True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
  
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))

class simpleAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1, activation='tanh')
    
    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size) 
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        
        # assert values.shape == (batch_size, max_len, hidden size)
        
        # we need to broadcast addition along the time axis to calculate the score
        
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(self.W1(query_with_time_axis)+self.W2(values))

        attention_weights = tf.nn.softmax(score, axis =1)

        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector, axis  = 1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.vocab_size= vocab_size
        self.decoder_units = decoder_units
        self.embedding = Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], trainable=False)
        self.gru = GRU(self.decoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        self.FC = Dense(self.vocab_size)
        self.attention = AttentionLayer(self.decoder_units)

    def call(self, x, hidden, encoder_output):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.FC(output)

        return x, state, attention_weights

class Generator(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, encoder_units, decoder_units,batch_size):
        super(Generator, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, encoder_units, batch_size)
        self.decoder = Decoder(vocab_size, embedding_dim, decoder_units, batch_size)
        self.batch_size = batch_size
        self.embedding_wts = self.encoder.embedding.get_weights()[0]

    def call(self, x, encoder_hidden):
        pass

    def approx_embedding(self, logits):
        """
        Logits are expected to be of shape : (bs, vocab_size) for every timestep
        We need the final embeddings after all timesteps like (bs, max_len, embedding_dim)
        the embedding_matrix is of shape vocab_size*embeddingdim
        """
        return tf.math.MatMul(logits, self.embedding_wts)