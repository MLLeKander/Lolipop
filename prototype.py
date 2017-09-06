#!/usr/bin/env python2
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras.layers import Layer, Dense
import numpy as np
import tensorflow as tf

class Prototype(Layer):
    def __init__(self, output_dim, input_dim=None, initial_protos=None, **kwargs):
        self.output_dim = output_dim
        self.initial_protos = initial_protos
        if input_dim:
            kwargs['input_shape'] = (input_dim,)
        assert len(kwargs.get('input_shape',(1,))) == 1
        super(Prototype, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(1,input_shape[1], self.output_dim),
                                 initializer='zero',
                                 name='{}_W'.format(self.name),
                                 trainable=True)
        if self.initial_protos is not None:
            self.set_protos(self.initial_protos)
        #del self.initial_protos
        super(Prototype, self).build(input_shape)  # Be sure to call this somewhere!

    def set_protos(self, new_protos):
        W_shape = K.get_value(self.W).shape
        new_shape = new_protos.shape
        if new_shape != W_shape:
            raise ValueError('Invalid shape. Expected %s but received %s'%(W_shape,new_shape))
        K.set_value(self.W, new_protos)

    def call(self, x, mask=None):
        # This may be slow or lead to OOM: http://stackoverflow.com/a/40053188
        x_ = K.expand_dims(x,-1)
        W_ = self.W
        out = K.sum((x_ - W_)**2, 1)
        print 'call(out):',out
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

class Linear(Dense):
    pass

def print_curves(history):
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy as np

    N = 51
    Nin, Nout = int(sys.argv[1]), int(sys.argv[2])

    def fill(arr):
        return np.column_stack((np.ones((arr.shape[0],Nin-1)),arr))

    Ps = np.array([.3, .6, 1][:Nout])
    initW = fill(np.array([.9, .1, 0.99][:Nout])).reshape(1,Nin,Nout)-0.1

    X = fill(np.linspace(0,10,N))
    Y = np.column_stack([(X[:,-1]-P)**2 for P in Ps])

    print 'Ps:',Ps
    print 'X.shape:',X.shape
    print 'Y.shape:',Y.shape


    model = Sequential([
        Prototype(output_dim=Y.shape[1], input_dim=X.shape[1]),#, initial_protos=initW),
    ])
    model.compile('rmsprop', 'mse', metrics=['accuracy'])
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)

    history = model.fit(X, Y, nb_epoch=10000, batch_size=N, verbose=0)
    scores = model.evaluate(X, Y)
    print '\n'
    for name, score in zip(model.metrics_names, scores):
        print('%s: %.2f' % (name, score))
    weight_names = [weight.name for layer in model.layers for weight in layer.weights]
    for name, weight in zip(weight_names, model.get_weights()):
        print name, '|', weight
    print model.predict(fill(Ps))

    print_curves(history)
