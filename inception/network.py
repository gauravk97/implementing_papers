# Residual attention56
# Gaurav Kumar
# May 9th, 2018

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, \
						 Concatenate, Add, Activation, BatchNormalization, Input, ZeroPadding2D, Dropout
from keras.models import Model


img_width = 224
img_height = 224
channels = 3

def inception_module(input_tensor, filters, name=""):
	path_1 = Conv2D(kernel_size=1, filters=filters[0], strides=1, padding="same", activation="relu")(input_tensor)
	path_2 = Conv2D(kernel_size=1, filters=filters[1], strides=1, padding="same", activation="relu")(input_tensor)
	path_2 = Conv2D(kernel_size=3, filters=filters[2], strides=1, padding="same", activation="relu")(path_2)
	path_3 = Conv2D(kernel_size=1, filters=filters[3], strides=1, padding="same", activation="relu")(input_tensor)
	path_3 = Conv2D(kernel_size=5, filters=filters[4], strides=1, padding="same", activation="relu")(path_3)
	path_4 = MaxPooling2D(pool_size=(3,3), strides=1, padding="same")(input_tensor)
	path_4 = Conv2D(kernel_size=1, filters=filters[5], strides=1, padding="same", activation="relu")(path_4)
	# path_4 = ZeroPadding2D(padding=(9))(path_4)
	out = Concatenate(name=name)([path_1, path_2, path_3, path_4])
	return out

input = Input(shape=(img_height, img_width, channels))
conv_1 = Conv2D(kernel_size=7, filters=64, strides=2, padding="same", activation="relu")(input)
pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(conv_1)
# LocalRespNorm
conv_2 = Conv2D(kernel_size=1, filters=64, strides=1, padding="same", activation="relu")(pool_1)
conv_3 = Conv2D(kernel_size=3, filters=192, strides=1, padding="same", activation="relu")(conv_2)
# LocalRespNorm
pool_2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(conv_3)

inception_1 = inception_module(pool_2, [64, 96, 128, 16, 32, 32], name="1")
inception_2 = inception_module(inception_1, [128, 128, 192, 32, 96, 64], name="2")

pool_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(inception_2)

inception_3 = inception_module(pool_3, [192, 96, 208, 16, 48, 64], name="2b")
inception_4 = inception_module(inception_3, [160, 112, 224, 24, 64, 64], name="3")
inception_5 = inception_module(inception_4, [128, 128, 256, 24, 64, 64], name="4")
inception_6 = inception_module(inception_5, [112, 144, 288, 32, 64, 64], name="5")
inception_7 = inception_module(inception_6, [256, 160, 320, 32, 128, 128], name="6")

pool_4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(inception_7)

inception_8 = inception_module(pool_4, [256, 160, 320, 32, 128, 128], name="7")
inception_9 = inception_module(inception_8, [384, 192, 384, 48, 128, 128], name="8")

pool_5 = AveragePooling2D((7,7), strides=(1,1), padding="valid")(inception_9)
dropout_1 = Dropout(0.4)(pool_5)
linear = Dense(1000)(dropout_1)
output = Activation("softmax")(linear)

model = Model(input=input, output=output)
model.summary()