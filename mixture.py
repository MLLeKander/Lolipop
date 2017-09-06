#!/usr/bin/env python2
from mylayers import *
from keras.layers import *
from keras.models import Model
from keras.callbacks import LambdaCallback as Callback
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from time import time

def error_curves(history):
    fig, ax1 = plt.subplots()
    x = range(1,len(history.history['acc'])+1)
    y1 = history.history['acc']
    ax1.plot(x, y1, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    y2 = history.history['loss']
    ax2.plot(x, y2, 'r.')
    ax2.set_ylabel('loss', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()

def print_weights(model):
    print '-----'
    weight_names = [weight.name for layer in model.layers for weight in layer.weights]
    for name, weight in zip(weight_names, model.get_weights()):
        print name, '|', weight
    print '-----'

def plot_model(*args,**kwargs):
    #print_weights(model)
    l2.set_ydata(model.predict(X))
    proto_locations =  K.get_value(model.get_layer('protos').W).flatten()#(-1,)
    l3.set_segments([np.column_stack(([x]*2, [-10,0])) for x in proto_locations])
    fig.canvas.draw()

N = 301
Nin = 1
Nout = 3

X = np.linspace(0,3,N)
if sys.argv[1] == 'step':
    Y = np.zeros((301,))
    Y[X<1] = -5
    Y[X>2] = -10
else:
    def f(x):
        return 2*x-1 if x < 1 else -x+2 if x < 2 else 3*x-6
    Y = np.vectorize(f)(X)
inputs = Input(shape=(Nin,), name='input')

kmeans = MiniBatchKMeans(Nout)
kmeans.fit(X.reshape(-1,1))

protos = Prototype(Nout, name='protos', protos=kmeans.cluster_centers_.reshape((1,3,1)))(inputs)
print 'protos:',protos
dist = Dist()(protos)
#dist = PrototypeDist(Nout, name='protos', protos=kmeans.cluster_centers_.reshape((1,3,1)))(inputs)

print 'dist:',dist
gate = ParameterizedExp(name='RBF',initial_beta=60)(dist)
print 'RBF:',gate
gate = L1Normalization(name='normalize')(gate)
print 'normalize:',gate

experts = Linear(name='experts')(protos)
#experts = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)), name='reshape')(protos)
#experts = Linear(name='experts')(experts)
#experts = Lambda(lambda x: K.squeeze(x,-1), name='reshape2')(experts)
print 'experts:',experts

outputs = merge([gate, experts], mode='dot', name='merge')
print 'outputs:',outputs


model = Model(input=inputs, output=outputs)

print 'expertWs:',model.get_layer('experts').W

# http://stackoverflow.com/questions/40074730/keras-mixture-of-models
def mix_loss(y_true, y_pred):
    o = model.get_layer('experts').output
    g = model.get_layer('normalize').output
    #return K.sum(g*K.square(y_true - o))
    #return -K.log(K.sum(g*K.exp(-0.5*K.square(y_true - o))))
    return -K.log(K.sum(g*K.exp(-0.5*K.abs(y_true - o))))
    #A = g[:, 0] * K.transpose(K.exp(-0.5 * K.square(y_true - o1)))
    #B = g[:, 1] * K.transpose(K.exp(-0.5 * K.square(y_true - o2)))
    #return -K.log(K.sum(A+B))
model.compile('rmsprop', loss='mse', metrics=['accuracy'])
#model.compile('rmsprop', loss=mix_loss, metrics=['accuracy'])
#model.compile('rmsprop', loss='mae', metrics=['accuracy'])

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=True)

fig, _ = plt.subplots()
l1, = plt.plot(X, Y, lw=2)
l2, = plt.plot(X, model.predict(X), lw=2)
l3 = plt.vlines(kmeans.cluster_centers_, [-10], [0])
plt.show(block=False)

raw_input()

t1 = time()
history = model.fit(X, Y, nb_epoch=2000, batch_size=20, verbose=0, callbacks=[Callback(on_batch_end=plot_model)])
t2 = time()
print t1-t2

scores = model.evaluate(X, Y)
print '\n'
for name, score in zip(model.metrics_names, scores):
    print('%s: %.2f' % (name, score))

error_curves(history)
