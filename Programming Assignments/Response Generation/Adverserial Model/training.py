# -*- coding: utf-8 -*-

# Data Pipeline
import os
try:
  import aesthetix as at
except:
  os.popen("pip install aesthetix")
  import aesthetix as at
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt 
print(tf.__version__)

# Downloading movie_lines
import os
if "movie_lines.txt" not in os.popen("ls").read():
  os.popen("wget https://raw.githubusercontent.com/Stellarator-X/ConvoBot/servus/Programming%20Assignments/Response%20Generation/movie_lines.txt")

movielines = open("movie_lines.txt", mode='r')
print(movielines)
lines = movielines.readlines()
print(len(lines))

def clean_str(_str):
  _str = _str.strip()
  _str = _str.lower()
  _str = _str.replace(".", "")
  _str = _str.replace(",", "")
  _str = _str.replace("?", "")
  _str = _str.replace("!", "")
  _str = _str.replace(":", "")
  _str = _str.replace(">", "")
  _str = _str.replace("<", "")
  _str = _str.replace("-", " ")
  _str = _str.replace("_", " ")
  _str = _str.replace("\\", "")
  _str = _str.replace("  ", " ")
  return _str

sample_size = 20000
cleanlines = []
for i, line in enumerate(lines[:sample_size]):
  at.progress_bar("Cleaning the lines", i, len(lines[:sample_size]))
  speaker, line = line.split('+++$+++ ')[-2:]
  cleanlines.append([speaker.split(" ")[0], line.split('\n')[0]])

cleanlines.reverse()
cleanlines = np.array(cleanlines)
for line in cleanlines[:10]:
  print(line[0],":",line[1])


# Forming the dataset 
response_data = []
l = len(cleanlines)-1
for i, line in enumerate(cleanlines[:-1]):
  at.progress_bar("Generating Stimulus-Response Pairs", i, l)
  speaker, utterance = line
  next_speaker, next_utterance = cleanlines[i+1]
  if speaker is not next_speaker:
    response_data.append(np.array(["<start> "+clean_str(utterance)+" <end>", "<start> "+clean_str(next_utterance)+" <end>"]))
  
response_data = np.array(response_data)
print(response_data.shape)
print(response_data[-10:])

oov_token = "<OOV>"
max_length = 25
stimuli = response_data[:, 0]
responses = response_data[:, 1]

tokenizer = Tokenizer(oov_token=oov_token, filters = "")
tokenizer.fit_on_texts(stimuli)
# with open("Tokens.txt") as file:
#   json_string = file.read()
# tokenizer = tokenizer_from_json(json_string)

word_index = tokenizer.word_index
index_word = {word_index[word]:word for word in word_index}
vocab_size = len(word_index)
stimulus_sequences = tokenizer.texts_to_sequences(stimuli)
response_sequences = tokenizer.texts_to_sequences(responses)

padded_stimulus_sequences = pad_sequences(stimulus_sequences, maxlen = max_length ,padding = 'post', truncating = 'post')
padded_response_sequences = pad_sequences(response_sequences, maxlen = max_length, padding = 'post', truncating = 'post')

start_token = word_index['<start>']

json_string = tokenizer.to_json()
with open("27_6_Tokens.txt", "w") as file:
  file.write(json_string)

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

print(encode_texts(['hello there', 'bye']))

print(decode_seq(encode_texts(['hello there', 'bye'])))

from sklearn.model_selection import train_test_split
X_train,  X_test, y_train,  y_test = train_test_split(padded_stimulus_sequences, padded_response_sequences, test_size = 0.1)

# Creating a tf dataset
BUFFER_SIZE = len(X_train)
BATCH_SIZE = 50
steps_per_epoch = (BUFFER_SIZE//BATCH_SIZE)
embedding_dim = 100
units = 1000

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)

"""# The Model"""

import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Getting the embedding layer
embedding_dim = 100
os.popen("wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \
    -O /tmp/glove.6B.100d.txt")
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
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

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.encoder_units = encoder_units
    self.batch_size = batch_size

    self.embedding = Embedding(vocab_size+1, embedding_dim, weights=[embeddings_matrix], trainable=False, mask_zero = True)
    self.gru = GRU(self.encoder_units, return_sequences=True, return_state = True)
    

  def call(self, X, hidden):
    x = self.embedding(X)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.encoder_units))

class simpleAttentionLayer(tf.keras.layers.Layer):
  
  def __init__(self, units):
    """
    params:
      units - units of the rnn layer preceeding this layer
    """
    super(simpleAttentionLayer, self).__init__()
    self.units = units
    self.W1 = Dense(units, name = "simple_attention_w1")    
    self.W2 = Dense(units, name = "simple_attention_w2")
    self.V = Dense(1, activation = 'tanh', name = 'simple_attention_v')

  def call(self, query, values):
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
    self.embedding = Embedding(vocab_size+1, embedding_dim, weights=[embeddings_matrix], trainable=False, mask_zero = True)
    self.gru = GRU(self.decoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.FC = Dense(self.vocab_size+1)
    self.attention = simpleAttentionLayer(self.decoder_units)

  def call(self, x, hidden, encoder_output):
    # context_vector, attention_weights = self.attention(hidden, encoder_output)
    x = self.embedding(x)
    # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.FC(output)
    return x, state#, attention_weights

class Generator(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, encoder_units, decoder_units,batch_size, max_len = max_length):
        super(Generator, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, encoder_units, batch_size)
        self.decoder = Decoder(vocab_size, embedding_dim, decoder_units, batch_size)
        self.batch_size = batch_size
        self.embedding_wts = tf.cast(embeddings_matrix,dtype = tf.float32)
        self.max_len = max_len

    def call(self, x, target): # Call for training
      # print(self.max_len, "is maxlen")
      encoder_hidden = self.encoder.initialize_hidden_state()
      decoder_hidden = encoder_hidden
      encoder_output, encoder_state = self.encoder(x, encoder_hidden)
      decoder_input = tf.expand_dims([start_token]*self.batch_size, 1)
      for t in range(self.max_len): 
          predictions, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
          decoder_input = tf.expand_dims(target[:, t], 1)
          if t is not 0:
            pred  = tf.concat([pred, tf.convert_to_tensor([predictions])],axis = 0)
          else:
            pred = tf.convert_to_tensor([predictions])
            # print("ex")
      try:
        pred = tf.transpose(pred, perm = [1,0,2])
      except:
        print(pred.shape)
        input()
      # pred = np.swapaxes(pred, 0, 1)
      app_emb = self.approx_embedding(pred)
      return app_emb

    def approx_embedding(self, logits):
        """
        Logits are expected to be of shape : (bs, vocab_size) for every timestep
        We need the final embeddings after all timesteps like (bs, max_len, embedding_dim)
        the embedding_matrix is of shape vocab_size*embeddingdim
        """
        logits = tf.cast(logits, dtype = tf.float32)
        
        return tf.linalg.matmul(logits, self.embedding_wts) # shape : (bs, embedding dim) for each timestep

class embeddingCNN(tf.keras.layers.Layer):

  def __init__(self, filters, kernel, **kwargs):
    super(embeddingCNN, self).__init__(**kwargs)
    self.c1 = Conv2D(filters, kernel, padding = 'same')
    self.c2 = Conv2D(1, kernel, padding = 'same')

  def call(self, X):
    return self.c2(self.c1(X))

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.embedding = Embedding(vocab_size+1, embedding_dim, weights=[embeddings_matrix], trainable=False)
    self.query_cnn = embeddingCNN(3, (2, 2), name = 'qcnn')
    self.response_cnn = embeddingCNN(3, (2, 2), name = 'rcnn')
    self.dense = Dense(100, activation='tanh')
    self.probability = Dense(1, activation = 'sigmoid')

  def pred_loss(self, true, gen):
    return tf.keras.losses.binary_crossentropy(true, gen)

  def call(self, input, target, generated_embeddings):
    inp_embed = tf.expand_dims(self.embedding(input), -1)
    tar_embed = tf.expand_dims(self.embedding(target), -1)
    generated_embeddings = tf.expand_dims(generated_embeddings, -1)
    A_q = self.query_cnn(inp_embed)
    A_R = self.response_cnn(tar_embed)
    A_r = self.response_cnn(generated_embeddings)

    R = Flatten()(tf.concat([A_q, A_R], axis = 0))
    r = Flatten()(tf.concat([A_q, A_r], axis = 0))
    R = self.dense(R)
    r = self.dense(r)
    p_R = self.probability(R)
    p_r = self.probability(r)
    one  = np.ones(p_R.shape)
    zero = np.zeros(p_r.shape)
    disc_loss = (tf.math.log(1-p_R+0.0000001)+tf.math.log(p_r+0.0000001))
    gen_loss = tf.math.abs(tar_embed - generated_embeddings)*100
    
    return disc_loss,gen_loss

"""# Training"""

generator = Generator(vocab_size, embedding_dim, units, units, BATCH_SIZE)

discriminator = Discriminator()

optimizer = tf.keras.optimizers.Adam()

# Pre Training Gen
gen_optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask
  mean =  tf.reduce_mean(loss)
  return mean

# Pre-training the Generator
@tf.function
def pre_train_step(inp, target, encoder_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    encoder_output, encoder_hidden = generator.encoder(inp, encoder_hidden)
    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([word_index['<start>']]*BATCH_SIZE, 1)

    for t in range(1, target.shape[1]): 
      predictions, decoder_hidden = generator.decoder(decoder_input, decoder_hidden, encoder_output)

      loss += loss_function(target[:, t], predictions)

      decoder_input = tf.expand_dims(target[:, t], 1)
    
  batch_loss = (loss/int(target.shape[1]))

  variables = generator.trainable_variables
  gradients = tape.gradient(loss, variables)
  gen_optimizer.apply_gradients(zip(gradients, variables))
  
  return batch_loss

import time
def pre_train_gen(sample_size, epochs, loss_hist = [], val_loss_hist = []):
  print(f"Training on {sample_size} samples : ")
  for epoch in range(1, epochs+1):
    start = time.time()
    encoder_hidden = generator.encoder.initialize_hidden_state()
    total_loss = disp_loss = val_loss = batch_loss = 0
    batch_elapsed = np.inf
    i=0
    eta = np.inf
    for batch, (inp, target) in enumerate(dataset.take(steps_per_epoch)):
      batch_start = time.time()
      elapsed = batch_start - start
      eta = (steps_per_epoch-(batch+1))*batch_elapsed
      at.progress_bar(f"Epoch {epoch:3.0f}/{epochs} ", batch, steps_per_epoch, output_vals = {'eta(s)':eta, 'time_elapsed(s)':elapsed, 'loss':disp_loss}, jump_line = True)
      batch_loss = pre_train_step(inp, target,encoder_hidden)
      total_loss += batch_loss
      disp_loss = total_loss/(max(1, batch))
      batch_elapsed = time.time()-batch_start
      
    # val_loss = batch_loss_func(val_y,val_x)
    # print(f"val_loss: {val_loss:3.4f}")
    total_loss = total_loss / steps_per_epoch
    loss_hist.append(total_loss)
  return loss_hist

lh = pre_train_gen(40000, 10)

# Pre-Training disc
@tf.function
def disc_pretrain_step(inp, tar):
  with tf.GradientTape() as tape:
    pred_embeddings = generator(inp, tar)
    d_loss, g_loss = discriminator(inp, tar, pred_embeddings)
    variables = discriminator.trainable_variables
    loss = d_loss
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
  return loss

def pretrain_disc(epochs = 2, loss_hist = []):
  print(f"Training on {sample_size} samples : ")
  for epoch in range(1, epochs+1):
    start = time.time()
    total_loss = disp_loss = val_loss = batch_loss = 0
    batch_elapsed = np.inf
    i=0
    eta = np.inf
    for batch, (inp, target) in enumerate(dataset.take(steps_per_epoch//2)):
      batch_start = time.time()
      elapsed = batch_start - start
      eta = (steps_per_epoch-(batch+1))*batch_elapsed
      at.progress_bar(f"Epoch {epoch:3.0f}/{epochs} ", batch, steps_per_epoch//2, output_vals = {'eta(s)':eta, 'time_elapsed(s)':elapsed, 'loss':disp_loss}, jump_line = True)
      batch_loss = disc_pretrain_step(inp, target)
      total_loss += tf.reduce_mean(batch_loss)
      disp_loss = total_loss/(max(1, batch))
      batch_elapsed = time.time()-batch_start
      
    # val_loss = batch_loss_func(val_y,val_x)
    # print(f"val_loss: {val_loss:3.4f}")
    total_loss = total_loss / steps_per_epoch
    loss_hist.append(total_loss)
  return loss_hist

# lh_d = pretrain_disc()
# plt.plot(lh_d)
# plt.show()

# @tf.function
def training_step(inp, tar, train_gen = True):
  with tf.GradientTape() as tape:
    pred_embeddings = generator(inp, tar)
    d_loss, g_loss = discriminator(inp, tar, pred_embeddings)

    # variables = generator.trainable_variables + discriminator.trainable_variables
    if train_gen:
      variables = generator.trainable_variables
      loss = g_loss
    else:
      variables = discriminator.trainable_variables
      loss = d_loss
    
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return g_loss, d_loss

# Training Loop
def fit_GAN(epochs = 10, change_every = 5,loss_hist = []):
  # Freezing some stuff
  for layer in generator.encoder.layers:
    layer.trainable = False

  d_loss = g_loss = 0
  train_gen = False
  for epoch in range(1, epochs+1):
    print(f"Epoch {epoch}/{epochs}")
    if epoch % change_every == 0:
      train_gen = not(train_gen)
    for batch, (inp, tar) in enumerate(dataset.take(steps_per_epoch)):
      G_loss, D_loss = training_step(inp, tar, train_gen)
      g_loss = (g_loss+tf.reduce_mean(G_loss))/(batch+1)
      d_loss = (d_loss+tf.reduce_mean(D_loss))/(batch+1)
      
      at.progress_bar(f"Batch {batch+1}/{steps_per_epoch}", batch, steps_per_epoch, output_vals = {'generator_loss':g_loss, 'discriminator_loss':d_loss})
    print()
    get_response_beam("who are you")
    if epoch % 5 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
  return loss_hist

# This is the latest version

"""# Evaluation"""

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
  return args

def beam_search_evaluate(sentence, beam_width=3, length_penalty_coef = 5, end_penalty_coef = 1):
  inputs = encode_texts([sentence])
  inputs = tf.convert_to_tensor(inputs)
  result = ['']*beam_width
  results = []
  scores_list = []
  scores = [0]*beam_width
  hidden = [tf.zeros((1, units))]
  encoder_output, encoder_hidden = generator.encoder(inputs, hidden)
  decoder_hidden = encoder_hidden
  decoder_input = tf.expand_dims([word_index['<start>']], 0)
  
  # At t = 0
  prediction, decoder_hidden = generator.decoder(decoder_input, decoder_hidden, encoder_output)
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
      pred, decoder_hidden[i] = generator.decoder(decoder_input[i], decoder_hidden[i], encoder_output)
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
      scores[i] = score.numpy()
      

    for i in range(beam_width):
      if index_word[predicted_id[i]]=='<end>':        
        results.append(result[i])
        norm_score = np.log(scores[i])
        length_penalty = ((5+t)/(5+1))**length_penalty_coef
        end_penalty = end_penalty_coef*len(sentence.split())/t
        norm_score = (norm_score/length_penalty) - end_penalty 
        scores_list.append(norm_score)
      
    
    if len(results) == beam_width:
      return results, scores_list
    
    for i in range(beam_width) : 
      decoder_input[i] = tf.expand_dims([predicted_id[i]], 0)

  return result, scores

def get_response_beam(sentence, beam_width=3):
  responses, scores = beam_search_evaluate(sentence, beam_width)
  print(f"Input : {sentence}")
  print("Responses")
  for r, s in zip(responses, scores):
    print(r.split(" <end>")[0], ": Score = ", s)

test_stimuli  = [
                 'Hello, I am abhay. What is your name?',
                 'who are you',
                 'Where are you going?',
                 'I\'m getting a little thirsty',
                 'Maybe you should go to the church today',
                 'I am  pretty sure that i am the dark lord',
                 'i am the dark lord',
                 'you should know about the bullets',
                 'I love you',
                 'why, are you scared?',
                 'are you scared?',
                 'I am not good at the talking part'
]

for stimulus in test_stimuli:
  get_response_beam(stimulus)
  print()

from google.colab import drive
drive.mount('/content/drive')

checkpoint_dir = '/content/drive/My Drive/Colab Notebooks/respGan_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, generator=generator, discriminator=discriminator)

checkpoint.save(file_prefix = checkpoint_prefix) # After training
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

index_word[0] = 'YADA'

loss_hist = fit_GAN(epochs = 20, change_every=4)

get_response_beam("Hello")

