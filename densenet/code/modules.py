import keras.backend as K
import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, Concatenate

def Dense_block(i, kernel_size, strides, filters):

    c1o = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides,\
        padding="same", activation="relu")(i)
    c2i = Concatenate()[i, c1o]
    c2o = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides,\
        padding="same", activation="relu")(c2i)
    c3i = Concatenate()[i, c1o, c2o]
    c3o = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides,\
        padding="same", activation="relu")(c3i)
    c4i = Concatenate()[i, c1o, c2o, c3o]
    c4o = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides,\
        padding="same", activation="relu")(c4i)
    return c4o

def Attention_block(i):
    

