import numpy as np

from keras.layers import Input, Dropout, MaxPooling2D, Conv2D, BatchNormalization, \
                        Activation, Concatenate, AveragePooling2D, Dense, Add, Flatten
from keras.models import Model

batch_size = 20
img_height = 224
img_width = 224
channels=3

def block_1(input_tensor, kernel_size, strides, filters):
    out = Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding="same")(input_tensor)
    out = Conv2D(filters=filters, strides=1, kernel_size=kernel_size, padding="same")(out)
    input_tensor = Conv2D(filters=filters, strides=strides, kernel_size=1, padding="same")(input_tensor)
    out = Add()([input_tensor, out])
    return out

def block_2(input_tensor, kernel_size, strides, filters):
	out = Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding="same")(input_tensor)
	out = Conv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding="same")(out)
	out = Add()([out, input_tensor])
	return out


## build model
input = Input(shape=(img_height, img_width, channels))
out = Conv2D(filters=64, strides=2, kernel_size=7, padding="same")(input)
out = MaxPooling2D((2,2))(out)

for _ in range(3):
	out = block_2(out, kernel_size=3, strides=1, filters=64)

out = block_1(out, kernel_size=3, strides=2, filters=128)

for _ in range(3):
	out = block_2(out, kernel_size=3, strides=1, filters=128)

out = block_1(out, kernel_size=3, strides=2, filters=256)

for _ in range(5):
	out = block_2(out, kernel_size=3, strides=1, filters=256)

out = block_1(out, kernel_size=3, strides=2, filters=512)

for _ in range(2):
	out = block_2(out, kernel_size=3, strides=1, filters=512)

out = AveragePooling2D(pool_size=(7,7))(out)
out = Flatten()(out)
output = Dense(1000)(out)

model = Model(input=input, output=output)
model.summary()