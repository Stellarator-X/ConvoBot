import numpy as np 
import tensorflow as tf 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Dense, Lambda, GRU, Bidirectional, Conv1D, Conv2D, TimeDistributed, Permute, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow_addons.seq2seq import BeamSearchDecoder
from ds_utils.layers import SeqWiseBatchNorm

"""
TODO 
    @stellarator-x
        beam search
    
    @anyone
        eos_index
        max_length
        lang model integration
"""

ALPHABET_LENGTH = 29
eos_index  = 29
max_length = 2850

class DSModel():

    def __init__(self, input_shape, alpha = 0.3, beta = 0.2, char_map={}):
        self.input_shape = input_shape
        self.char_map = char_map
        self.idx_map = {char_map[ch]:ch for ch in char_map}
        # Tunable hyperparams for net_loss
        self.alpha = alpha
        self.beta = beta
        
        
    
    def build(self, Name = "DeepSpeech2", num_conv = 3, num_rnn = 7, beam_width = 50):
        
        self.model =  Sequential(name = Name)
        self.model.add(Input(shape = self.input_shape))
        
        if(len(self.input_shape)==2):
            self.model.add(Lambda(lambda x : tf.expand_dims(x, axis = -1)))

        # Conv Layers
        self.model.add(Conv2D(filters = 16, kernel_size = (3, 3), strides = 3, padding='same',  name = f"Conv1"))
        for i in range(1, num_conv):
            self.model.add(Conv2D(filters = 16, kernel_size = (3, 3), strides = 3, padding='same',name = f"Conv{i+1}"))
        
        # Conv2RNN 
        self.model.add(Reshape((38*75, 16)))

        # RNN Layers
        for i in range(num_rnn):
            self.model.add(Bidirectional(GRU(units = 800, return_sequences=True), name = f"RNN{i+1}")),
            self.model.add(SeqWiseBatchNorm(name = f"BatchNorm{i+1}"))
        
        # Final Layer
        self.model.add(TimeDistributed(Dense(units = ALPHABET_LENGTH, activation='softmax'), name = "OutputLayer"))

        try:
            return self.model
        except:
            print("Couldn't build the model")
            return

    def get_label(self, y_pred):
        label = ""
        for row in y_pred.numpy():
            idx = K.argmax(row)
            idx=idx.numpy()
            try:
                ch = self.idx_map[idx]
            except TypeError as e:
                print(e)
                print(idx.shape)
                return
            if ch is not "_":label+=ch
        return label
    
    @tf.function # TODO : see why the decorator isn't working
    def ctc_find_eos(self, y_true, y_pred):
        # From SO : Todo : var init, predlength objective
        # convert y_pred from one-hot to label indices
        y_pred_ind = K.argmax(y_pred, axis=-1)

        #to make sure y_pred has one end_of_sentence (to avoid errors)
        y_pred_end = K.concatenate([y_pred_ind[:,:-1], eos_index * K.ones_like(y_pred_ind[:,-1:])], axis = 1)

        #to make sure the first occurrence of the char is more important than subsequent ones
        occurrence_weights = K.arange(start = 0, stop=max_length, dtype=K.floatx())

        is_eos_true = K.cast_to_floatx(K.equal(y_true, eos_index))
        is_eos_pred = K.cast_to_floatx(K.equal(y_pred_end, eos_index))

        #lengths
        true_lengths = 1 + K.argmax(occurrence_weights * is_eos_true, axis=1)
        pred_lengths = 1 + K.argmax(occurrence_weights * is_eos_pred, axis=1)

        #reshape
        true_lengths = K.reshape(true_lengths, (-1,1))
        pred_lengths = K.reshape(pred_lengths, (-1,1))

        return K.ctc_batch_cost(y_true, y_pred, pred_lengths, true_lengths)# + self.beta(pred_lengths) # Maybe a temp fix
        # y_pred_ind = tf.cast(tf.expand_dims(K.argmax(y_pred, axis = -1), -1), tf.float64)

        # label_length = tf.cast(tf.convert_to_tensor(np.array([y_true.shape[-1]]*y_true.shape[1])), dtype = tf.int32)
        # logit_length = tf.cast(tf.convert_to_tensor(np.array([2850]*y_pred.shape[1])), dtype = tf.int32)
        # ctcloss = tf.nn.ctc_loss(y_true, y_pred_ind, label_length, logit_length, logits_time_major=False)
        # return ctcloss

    @staticmethod
    def net_loss(y_true, y_pred):
        # Summation log loss with ctc, word_count, lang model
        # Q(y) = log(p ctc (y|x)) + α log(p lm (y)) + β word_count(y)
        Loss = K.log(ctc_find_eos(y_true, y_pred)) #+ self.beta*word_count(y_pred) : need obviated with temp fix
        return Loss

    def summary(self):
        self.model.summary()

    def compile(self):
        return self.model.compile(loss  = self.ctc_find_eos, optimizer = 'adam', metrics = ['accuracy'])

    def fit(self, **kwargs):
        print(kwargs)
        return self.model.fit(**kwargs)

    def getModel(self):
        return self.model
    
    def prediction(self, X):
        pred = []
        for x in X.numpy():
            pred.append(self.get_label(x))
        return np.array(pred)
