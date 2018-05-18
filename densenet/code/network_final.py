### Densenet 121

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, Concatenate, \
						Add, Dense, AveragePooling2D
from keras.models import Model

img_width = 224
img_height = 224
channels = 3
k=32 # growth rate

def conv_block():
	pass

def transition_block():
	pass

def dense_module(input_tensor, filters, number_of_units, name):
	concat = input_tensor
	for x in range(number_of_units):
		out = Conv2D(filters=4*k, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(concat)
		out = Conv2D(filters=filters[x], kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(out)
		concat = Concatenate(name=name+"_"+str(x))([concat, out])

	return concat


input = Input((img_height, img_width, channels))
conv_1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation="relu")(input)
pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(conv_1)

dense_1 = dense_module(pool_1, [35, 67, 99, 131, 163], 6, "dense_1")

conv_2 = Conv2D(filters=81, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(dense_1)
pool_2 = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_2)

dense_2 = dense_module(pool_2, [], 12, "dense_2")

conv_3 = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(dense_2)
pool_3 = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_3)

dense_3 = dense_module(pool_3, filters, 24, "dense_3")

conv_4 = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(dense_3)
pool_4 = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")(conv_4)

dense_4 = dense_module(pool_4, filters, 16, "dense_4")

pool_5 = AveragePooling2D(pool_size=(7,7), strides=(1,1), padding="valid")(dense_4)
output = Dense(1000, activation="softmax")(pool_5)

model = Model(input=input, output=output)
model.summary()