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
eos_index  = 28
max_length = 100

class DSModel():

    def __init__(self, input_shape, alpha = 0.3, beta = 0.2, char_map={}):
        self.input_shape = input_shape
        self.char_map = char_map
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
        self.model.add(Reshape((256)))

        # RNN Layers
        for i in range(num_rnn):
            self.model.add(Bidirectional(GRU(units = 800, return_sequences=True), name = f"RNN{i+1}")),
            self.model.add(SeqWiseBatchNorm(name = f"BatchNorm{i+1}"))

        # Beam Search Layer : For later implementations, requires a monodir rnn cell
        # self.model.add(BeamSearchDecoder(
        #     cell = self.model.get_layer(name = f"RNN{i+1}"),
        #     beam_width = beam_width,
        #     output_layer = Dense(800)
        # ))
        
        # Final Layer
        self.model.add(TimeDistributed(Dense(units = ALPHABET_LENGTH, activation='softmax'), name = "OutputLayer"))

        try:
            return self.model
        except:
            print("Couldn't build the model")
            return

    def get_label(y_pred):
        
        pass

    def ctc_find_eos(y_true, y_pred):
        # From SO : Todo : var init, predlength objective
        #convert y_pred from one-hot to label indices
        y_pred_ind = K.argmax(y_pred, axis=-1)

        #to make sure y_pred has one end_of_sentence (to avoid errors)
        y_pred_end = K.concatenate([y_pred_ind[:,:-1], eos_index * K.ones_like(y_pred_ind[:,-1:])], axis = 1)

        #to make sure the first occurrence of the char is more important than subsequent ones
        occurrence_weights = K.arange(start = max_length, stop=0, dtype=K.floatx())

        is_eos_true = K.cast_to_floatx(K.equal(y_true, eos_index))
        is_eos_pred = K.cast_to_floatx(K.equal(y_pred_end, eos_index))

        #lengths
        true_lengths = 1 + K.argmax(occurrence_weights * is_eos_true, axis=1)
        pred_lengths = 1 + K.argmax(occurrence_weights * is_eos_pred, axis=1)

        #reshape
        true_lengths = K.reshape(true_lengths, (-1,1))
        pred_lengths = K.reshape(pred_lengths, (-1,1))

        return K.ctc_batch_cost(y_true, y_pred, pred_lengths, true_lengths) + self.beta(pred_lengths) # Maybe a temp fix

    @staticmethod
    def net_loss(y_true, y_pred):
        # Summation log loss with ctc, word_count, lang model
        # Q(y) = log(p ctc (y|x)) + α log(p lm (y)) + β word_count(y)
        Loss = K.log(ctc_find_eos(y_true, y_pred)) + self.alpha*K.log(LangMod(y_pred)) #+ self.beta*word_count(y_pred) : need obviated with temp fix
        return Loss

    def summary(self):
        self.model.summary()

    def compile(self):
        return self.model.compile(loss  = self.net_loss, optimizer = 'adam', metrics = ['word_error_rate'])


    def fit(self, **kwargs):
        print(kwargs)
        return self.model.fit(**kwargs)

    def getModel(self):
        return self.model

if __name__ == "__main__":
    mod = DSModel((1025, 2000))
    mod.build()
    mod.compile()
    print(mod.model.summary())
    pred = mod.model.predict(np.array([np.random.randn(1025, 2000)]))
    print(pred.shape)
