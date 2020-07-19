print("Loading dependencies ...")

import os
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Dense   

def clean_str(_str):
  _str = _str.strip()
  _str = _str.lower()
  _str = _str.replace(".", "")
  _str = _str.replace(",", "")
  _str = _str.replace("!", "")
  _str = _str.replace(":", "")
  _str = _str.replace("-", " ")
  _str = _str.replace("_", " ")
  _str = _str.replace("\\", "")
  _str = _str.replace("  ", " ")
  return _str

with open("/home/abhay/Projects/ConvoBot/Programming Assignments/Response Generation/bin/Tokens.txt", 'r') as file:
    js_string = file.read()
    tokenizer = tokenizer_from_json(js_string)

word_index = tokenizer.word_index
index_word = {word_index[word]:word for word in word_index}
vocab_size = len(word_index)
max_length = 25

def encode_texts(str_list, tokenizer=tokenizer, max_length = max_length):
  # print(str_list)
  str_list = ["<start> " + s + " <end>" for s in str_list]
  seq = tokenizer.texts_to_sequences(str_list)
  pad_seq = pad_sequences(seq, max_length, padding  ='post', truncating='post')
  return pad_seq

def decode_seq(seq_list, tokenizer = tokenizer):
  ret = tokenizer.sequences_to_texts(seq_list)
  ret = [s.split("<start>")[1].split("<end>")[0][1:-1] for s in ret]
  return np.array(ret)

embedding_dim = 100
units = 1000
BATCH_SIZE = 100

# Getting the trained GloVe embeddings

# os.system("wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt -O bin/glove.6B.100d.txt")
embeddings_index = {};


with open('/home/abhay/Projects/ConvoBot/Programming Assignments/Response Generation/bin/glove.6B.100d.txt') as f:
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


# The encoder and Decoder Models

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

encoder = Encoder(vocab_size+1, embedding_dim, units, BATCH_SIZE)
sagetmple_hidden = encoder.initialize_hidden_state()
# sample_out, sample_hidden = encoder(example_input_batch,sample_hidden)

class AttentionLayer(tf.keras.layers.Layer):
  
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
    
attention_layer = AttentionLayer(10)
# attention_result, attention_weights = attention_layer(sample_hidden, sample_out)


# The decoder model

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
  
decoder = Decoder(vocab_size+1, embedding_dim, units, BATCH_SIZE)

# sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_out)

# Defining the optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask
  return tf.reduce_mean(loss)

# Defining checkpoint variables
checkpoint_dir = '/home/abhay/Projects/ConvoBot/Programming Assignments/Response Generation/bin/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

def evaluate(sentence):
  attention_plot = np.zeros((max_length, max_length))

  inputs = encode_texts([sentence])
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]

  encoder_output, encoder_hidden = encoder(inputs, hidden)

  decoder_hidden = encoder_hidden
  decoder_input = tf.expand_dims([word_index['<start>']], 0)

  for _ in range(max_length):
    predictions, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)

    # Storing the attention weights to plot later
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()
    # print(predictions.shape)
    # input()
    predicted_id = tf.argmax(predictions[0]).numpy()

    result += index_word[predicted_id]+' '

    if index_word[predicted_id]=='<end>':
      return result, sentence, attention_plot
    
    decoder_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# Plotting attention weights
import matplotlib.ticker as ticker
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()

def get_response(sentence, plot_graph = False):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted Response: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  if plot_graph :plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def gresp(sentence):
    result, _, _ = evaluate(sentence)
    return result

def argmax_beam(tensor, width):
  arr = tensor.numpy()
  # assert arr.shape[0] == 1
  arr_ = [c for c in arr[0]]
  # assert len(arr_) >= width, "Beam width is greater than the tensor length"
  args = []
  for i in range(width):
    argm = np.argmax(arr_)
    args.append(argm)
    arr_[argm] = -np.inf
  # print(args)
  # input()
  return args

def beam_search_evaluate(sentence, beam_width=3, length_norm_alpha = 0.5):
  inputs = encode_texts([sentence])
  inputs = tf.convert_to_tensor(inputs)
  result = ['']*beam_width
  results = []
  scores_list = []
  scores = [0]*beam_width
  hidden = [tf.zeros((1, units))]
  encoder_output, encoder_hidden = encoder(inputs, hidden)
  decoder_hidden = encoder_hidden
  decoder_input = tf.expand_dims([word_index['<start>']], 0)
  
  # At t = 0
  prediction, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
  ids = argmax_beam(prediction, beam_width)
  predicted_id = []
  for i, id in enumerate(ids):
    result[i]+= index_word[id] + " "
    predicted_id.append(id)
  decoder_input = []
  for i in range(beam_width) : 
      decoder_input.append(tf.expand_dims([predicted_id[i]], 0))
  decoder_hidden = [decoder_hidden]*beam_width

  # After t=0  
  for t in range(1, max_length):
    predictions = []
    
    for i in range(beam_width):
      pred, decoder_hidden[i], _ = decoder(decoder_input[i], decoder_hidden[i], encoder_output)
      predictions.append(pred)
    ids = [] # List of tuples : (id, predicted probability, decoder_hidden from beam)
    
    for beam in range(beam_width):
      predicted_ids = argmax_beam(predictions[beam], beam_width)
      for id in predicted_ids:
        ids.append((id, predictions[beam][0,id],decoder_hidden[beam], result[beam])) 
    
    probs = tf.convert_to_tensor([[i[1] for i in ids]])
    best_id_indices = argmax_beam(probs, beam_width)
    predicted_id = []

    for i, idx in enumerate(best_id_indices):
      pr_id, score, decoder_hidden[i], result[i] = ids[idx]
      result[i] += index_word[pr_id] + " "
      predicted_id.append(pr_id)
      scores[i] = np.log(score.numpy())
      

    for i in range(beam_width):
      if index_word[predicted_id[i]]=='<end>':        
        results.append(result[i]) 
        # scores_list.append(scores[i])#*np.log(t))
        
    
    if len(results) == beam_width:
      return results, scores
    
    for i in range(beam_width) : 
      decoder_input[i] = tf.expand_dims([predicted_id[i]], 0)

  return result, scores

def get_response_beam(sentence, beam_width=3):
  responses, scores = beam_search_evaluate(sentence, beam_width)
  r = responses[np.argmax(scores)]
  return r.split(" <end>")[0]


# Restore from last checkpoint 
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
