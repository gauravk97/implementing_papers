import keras.backend as K
import tensorflow as tf
import os
import numpy as np

from keras.layers import Input, Dropout, AveragePooling2D, Conv2D
from keras.models import Model
from modules import Attention_block, Dense_block, CTC_block

batch_size = 20
img_height = 32
img_width = 32

class Network(object):
    def __init__(self, batch_size, img_height, img,_width):
        self.batch_size=batch_size
        self.height = img_height
        self.width = img_width


    def network():
        i = Input((self.height, self.width, 1))
        # Encoder        
        x = Conv2D(kernel_size=(3,3), filters=36, strides=(1,1), padding="same", \
            activation="relu")(i)

        x = Dense_block(x, kernel_size=(3,3), strides=(1,1), filters=36)
        x = Attention_block()(x)
        x = AveragePooling2D((2,2))(x)

        x = Dense_block(x, kernel_size=(3,3), strides=(1,1), filters=36)
        x = Attention_block()(x)
        x = AveragePooling2D((2,2))(x)

        x = Dense_block(x, kernel_size=(3,3), strides=(1,1), filters=36)
        x = Conv2D(kernel_size=(3,3), filters=512, strides=(1,1), padding="same", \
            activation="relu")
        x = AveragePooling2D((2,2))(x)
        x = Conv2D(kernel_size=(3,3), filters=512, strides=(1,1), padding="same", \
            activation="relu")
        # CNN
        x = Conv2D(kernel_size=(3,3), filters=1, strides=(2,1), padding="same", \
            activation="relu")(x)
        x = Conv2D(kernel_size=(3,3), filters=1, strides=(2,1), padding="same", \
            activation="relu")(x)
        x = Conv2D(kernel_size=(3,3), filters=1, strides=(2,1), padding="same", \
            activation="relu")(x)
        x = Conv2D(kernel_size=(3,3), filters=1, strides=(2,1), padding="same", \
            activation="relu")(x)
        # CTC
        x = CTC()(x)
        
        self.Model = Model(i, x)        
    
    def train():
        


