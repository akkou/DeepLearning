# 
import numpy as np
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
#model = gensim.models.Word2Vec(sentences, min_count=1)
np.version.version

from StringIO import StringIO
test = "a,1,2\nb,3,4"
a = np.genfromtxt(StringIO(test), delimiter=",", dtype=None)
a.ndim

query_lal_x = np.genfromtxt("/Users/naver/Documents/map/data/embed_lal.txt", dtype = None, delimiter="\t")
                        # skip_header=0)
                        
import pandas as pd
from pandas import DataFrame

myDF = DataFrame(query_lal_x)
query_lal_x = myDF

query = query_lal_x['f0']

lal = myDF.loc[:,'f1':'f2']
#lal = np.hstack((query_lal_x['f1'], query_lal_x['f2']))
lal.shape

import math
math.floor(98027/64)*64



x = query_lal_x.loc[0:97983,'f3':'f102']

# for i in fields_all[3:103]:
#    x = np.hstack((x,query_lal_x[i]))

x = x.values
x.shape

import keras.backend as K
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model

m = 64
n_z = 2
n_epoch = 50

X    = Input(batch_shape=(m, 100))
cond = Input(batch_shape=(m, 2))

# Q(z|X) -- encoder
inputs = concatenate([X, cond], axis=1) # not using in simple vae

h_q = Dense(60, activation='relu')(X)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X,y)
z = Lambda(sample_z)([mu, log_sigma])

# P(X|z,y) -- decoder
decoder_hidden = Dense(60,  activation='relu')
decoder_out    = Dense(100, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# Overall VAE model, for reconstruction and training
vae = Model(X, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(X, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in) # error
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

from keras import losses

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z,y)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis = -1)
    # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl
    
vae.compile(optimizer='adam', loss=vae_loss)
# fit/Train
vae.fit(x, x, batch_size=m, epochs=n_epoch)

vae.save("./vae.h5")

# https://blog.keras.io/building-autoencoders-in-keras.html
x_test_encoded = encoder.predict(x, batch_size=m)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test) # mnist example
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s = .3)
#plt.colorbar()
plt.show()

