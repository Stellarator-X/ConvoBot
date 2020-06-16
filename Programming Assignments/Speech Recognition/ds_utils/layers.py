import numpy as np 
import tensorflow as tf
import copy


class SeqWiseBatchNorm(tf.keras.layers.Layer):

    def __init__(self, return_sequences = True, **kwargs):
        """
        param : return_sequences - return_sequences equivalent for the previous layer;
                        We need the previous layers' return_sequences to be True for seq_wise batchnorm
        """
       

        # For input-dependent initialisation
        super(SeqWiseBatchNorm, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # Rest of the initialisation: learnable params etc, now that we know the shape of the inputs
        self.Lambda = self.add_weight(shape= (), initializer = 'ones', trainable = True)
        self.beta = self.add_weight(shape= (), initializer = 'zeros', trainable = True)
        self.eps = 1e-3

    def call(self, input):
        # 'Forward' computation
        mean, var = tf.nn.moments(input, axes = [0, 1], keepdims = True)
        result = (input - mean)/tf.math.sqrt(var+self.eps)
        result = self.Lambda*result + self.beta
        if not self.return_sequences:
            result[:, -1, :]
        return result

class RowConv1D(tf.keras.layers.Layer):
    # TODO

    def __init__(self, window_size, strides = 1):
        print("init 1")
        super(RowConv1D, self).__init__()
        self.window_size = window_size
        self.strides = strides

    def build(self, input_shape):
        self.W = self.add_weight(shape = (input_shape[-1],self.window_size), initializer = 'random_normal', trainable = True)
                
    def call(self, input):
        # Naive, Bugged, TODO@stellarator-x:
        result = []
        d = input.shape[-1]
        s = self.window_size
        m = input.shape[-2]
        for i in input:
            I = tf.reshape(i, [1,d,1,m])
            W = tf.reshape(self.W, [1,d,s,1])
            strides = [1,self.strides,1,1]
            out = tf.nn.depthwise_conv2d(I, W, strides)
            result.append(out)
        tf.convert_to_tensor(result)
        return result