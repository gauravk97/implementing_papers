# Residual attention56
# Gaurav Kumar
# May 9th, 2018

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, \
						 Concatenate, Add, Activation, BatchNormalization, Input, ZeroPadding2D, Dropout, Multiply
from keras.models import Model


img_width = 224
img_height = 224
channels = 3

def residual_module(input_tensor, filters):
	out = Conv2D(filters=filters[0], kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(input_tensor)
	out = Conv2D(filters=filters[1], kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(out)
	out = Conv2D(filters=filters[2], kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(out)
	input_tensor = Conv2D(filters=filters[2], kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(input_tensor)
	out = Add()([input_tensor, out])
	return out

def attention_module(input_tensor, filters):
	att_res_1 = residual_module(input_tensor, filters)
	
	trunk_res_1 = residual_module(att_res_1, filters)
	
	mask_pool_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding="same")(att_res_1)
	mask_res_1 = residual_module(mask_pool_1, filters)
	mask_pool_2 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding="same")(mask_res_1)
	mask_res_2 = residual_module(mask_pool_2, filters)
	mask_res_3 = residual_module(mask_res_2, filters)
	#interpolation
	mask_res_res = residual_module(mask_res_1, filters)
	mask_add = Add()([mask_res_res, mask_res_3])
	mask_res_4 = residual_module(mask_add ,filters)
	# interpolation
	mask_conv_1 = Conv2D(filters=filters[0], kernel_size=1, strides=1, activation="relu", padding="same")(mask_res_4)
	mask_conv_2 = Conv2D(filters=filters[0], kernel_size=1, strides=1, activation="relu", padding="same")(mask_conv_1)
	mask_act = Activation("sigmoid")(mask_conv_2)

	att_mul = Multiply()([mask_act, trunk_res_1])
	att_out = Add()([trunk_res_1, att_mul])
	# att_out = (1+mask_act)*trunk_res_1

	att_res_2 = residual_module(att_out, filters)
	return att_res_2



input = Input(shape=(img_height, img_width, channels))
conv_1 = Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", padding="same")(input)
pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(conv_1)
res_1 = residual_module(pool_1, [64, 64, 256])
attention_1 = attention_module(res_1, [256, 256, 256])
pool_2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(attention_1)
res_2 = residual_module(pool_2, [128, 128, 512])
attention_2 = attention_module(res_2, [512, 512, 512])
pool_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(attention_2)
res_3 = residual_module(pool_3, [256, 256, 1024])
attention_3 = attention_module(res_3, [1024, 1024, 1024])
pool_4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(attention_3)
res_4 = residual_module(pool_4, [512, 512, 2048])
pool_5 = AveragePooling2D(pool_size=(7,7), strides=(1,1))(res_4)
linear = Dense(1000)(pool_5)
output = Activation("softmax")(linear)

model = Model(input=input, output=output)
model.summary()