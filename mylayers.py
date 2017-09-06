#!/usr/bin/env python
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras.layers import Layer, Dense
from keras.constraints import MaxNorm
import numpy as np
import tensorflow as tf

class PrototypeDist(Layer):
    def __init__(self, output_dim, input_dim=None, protos=None, **kwargs):
        self.output_dim = output_dim
        self.initial_protos = protos
        if input_dim:
            kwargs['input_shape'] = (input_dim,)
        assert len(kwargs.get('input_shape',(1,))) == 1
        super(PrototypeDist, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(1, self.output_dim, input_shape[1]),
                initializer='zero', name='%s_W'%self.name)
        if self.initial_protos is not None:
            self.set_protos(self.initial_protos)
        #del self.initial_protos
        super(PrototypeDist, self).build(input_shape)  # Be sure to call this somewhere!

    def set_protos(self, new_protos):
        W_shape = K.get_value(self.W).shape
        new_shape = new_protos.shape
        if new_shape != W_shape:
            raise ValueError('Invalid shape. Expected %s but received %s'%(W_shape,new_shape))
        K.set_value(self.W, new_protos)

    def call(self, x, mask=None):
        # This may be slow or lead to OOM: http://stackoverflow.com/a/40053188
        x_ = K.expand_dims(x,1)
        W_ = self.W
        out = K.sum((x_ - W_)**2, 2)
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

class Prototype(Layer):
    def __init__(self, num_protos, input_dim=None, protos=None, **kwargs):
        self.num_protos = num_protos
        self.initial_protos = protos
        if input_dim:
            kwargs['input_shape'] = (input_dim,)
        assert len(kwargs.get('input_shape',(1,))) == 1
        super(Prototype, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(1, self.num_protos, input_shape[1]),
                initializer='zero', name='%s_W'%self.name)
        if self.initial_protos is not None:
            self.set_protos(self.initial_protos)
        #del self.initial_protos
        super(Prototype, self).build(input_shape)

    def set_protos(self, new_protos):
        W_shape = K.get_value(self.W).shape
        new_shape = new_protos.shape
        if new_shape != W_shape:
            raise ValueError('Invalid shape. Expected %s but received %s'%(W_shape,new_shape))
        K.set_value(self.W, new_protos)

    def call(self, x, mask=None):
        # This may be slow or lead to OOM: http://stackoverflow.com/a/40053188
        x_ = K.expand_dims(x,1)
        W_ = self.W
        out = x_ - W_
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_protos, input_shape[1])

class Dist(Layer):
    def call(self, x, mask=None):
        return K.sum(x**2, 2)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

class ParameterizedExp(Layer):
    def __init__(self, initial_beta=1., **kwargs):
        self.initial_beta = initial_beta
        super(ParameterizedExp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(shape=(1,), initializer='zero', name='%s_beta'%self.name,constraint=MaxNorm(60))
        if self.initial_beta is not None:
            print((np.array(self.initial_beta),))
            self.set_weights((np.array(self.initial_beta).reshape((1,)),))
        super(ParameterizedExp, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return K.exp(-self.beta*x)

class WinnerTakeAll(Layer):
    def call(self, x, mask=None):
        pass

class L1Normalization(Layer):
    def call(self, x, mask=None):
        return x/K.sum(x,1,True)

class Linear(Layer):
    def __init__(self, weights=None, b=None, **kwargs):
        self.initial_weights = weights
        self.initial_b = b
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(1,) + input_shape[1:],
                initializer='zero', name='%s_W'%self.name)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

        self.b = self.add_weight(shape=(1,input_shape[1]),
                initializer='zero', name='%s_b'%self.name)
        if self.initial_b is not None:
            self.set_b(self.initial_b)

        super(Linear, self).build(input_shape)

    def call(self, x, mask=None):
        return K.sum(tf.multiply(x, self.W), 2) + self.b

    def get_output_shape_for(self, input_shape):
        return input_shape[:2]

    def set_weights(self, new_weights):
        W_shape = K.get_value(self.W).shape
        new_shape = new_weights.shape
        if new_shape != W_shape:
            raise ValueError('Invalid shape. Expected %s but received %s'%(W_shape,new_shape))
        K.set_value(self.W, new_weights)

    def set_b(self, new_b):
        K.set_value(self.b, np.array(new_b).reshape((1,-1)))
