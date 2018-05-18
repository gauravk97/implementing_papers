# DenseNet-121 
# May 7th, 2018
# Gaurav Kumar

import numpy as np

from keras.layers import Input, Dropout, MaxPooling2D, Conv2D, BatchNormalization, \
                        Activation, Concatenate, AveragePooling2D, Dense
from keras.models import Model

batch_size = 20
img_height = 224
img_width = 224
channels=1

ip = Input(shape=(img_height, img_width, channels))
c = Conv2D(filters=32, kernel_size=(7,7), strides=(2,2), padding='same')(ip)
m = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(c)

# dense block 1 starts
bn_1_1 = BatchNormalization()(m)
act_1_1 = Activation("relu")(bn_1_1)
c_1_1 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_1_1)

bn_1_2 = BatchNormalization()(c_1_1)
act_1_2 = Activation("relu")(bn_1_2)
c_1_2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_2)

concat_1_1 = Concatenate()([c_1_1, c_1_2])

bn_1_3  = BatchNormalization()(concat_1_1)
act_1_3 = Activation("relu")(bn_1_3)
c_1_3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_1_3)

concat_1_2 = Concatenate()([c_1_1, c_1_2, c_1_3])

bn_1_4  = BatchNormalization()(concat_1_2)
act_1_4 = Activation("relu")(bn_1_4)
c_1_4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_4)

concat_1_3 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4])

bn_1_5  = BatchNormalization()(concat_1_3)
act_1_5 = Activation("relu")(bn_1_5)
c_1_5 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_1_5)

concat_1_4 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5])

bn_1_6  = BatchNormalization()(concat_1_4)
act_1_6 = Activation("relu")(bn_1_6)
c_1_6 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_6)

concat_1_5 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6])

bn_1_7  = BatchNormalization()(concat_1_5)
act_1_7 = Activation("relu")(bn_1_7)
c_1_7 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_7)

concat_1_6 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7])

bn_1_8  = BatchNormalization()(concat_1_6)
act_1_8 = Activation("relu")(bn_1_8)
c_1_8 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_8)

concat_1_7 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8])

bn_1_9  = BatchNormalization()(concat_1_7)
act_1_9 = Activation("relu")(bn_1_9)
c_1_9 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_9)

concat_1_8 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8, c_1_9])

bn_1_10  = BatchNormalization()(concat_1_8)
act_1_10 = Activation("relu")(bn_1_10)
c_1_10 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_10)

concat_1_9 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8, c_1_9, c_1_10])

bn_1_11  = BatchNormalization()(concat_1_9)
act_1_11 = Activation("relu")(bn_1_11)
c_1_11 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_11)

concat_1_10 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8, c_1_9, c_1_10, c_1_11])

bn_1_12  = BatchNormalization()(concat_1_10)
act_1_12 = Activation("relu")(bn_1_12)
c_1_12 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_1_12)

concat_1_11 = Concatenate()([c_1_1, c_1_2, c_1_3, c_1_4, c_1_5, c_1_6, c_1_7, c_1_8, c_1_9, c_1_10, c_1_11, c_1_12])

### dense block 1 ends

conv2 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(concat_1_11)
avg_pool_1 = AveragePooling2D(pool_size=(2,2), strides=(2,2))(conv2)

### dense 2
bn_2_1 = BatchNormalization()(avg_pool_1)
act_2_1 = Activation("relu")(bn_2_1)
c_2_1 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_2_1)

bn_2_2 = BatchNormalization()(c_2_1)
act_2_2 = Activation("relu")(bn_2_2)
c_2_2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_2)

concat_2_1 = Concatenate()([c_2_1, c_2_2])

bn_2_3  = BatchNormalization()(concat_2_1)
act_2_3 = Activation("relu")(bn_2_3)
c_2_3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_2_3)

concat_2_2 = Concatenate()([c_2_1, c_2_2, c_2_3])

bn_2_4  = BatchNormalization()(concat_2_2)
act_2_4 = Activation("relu")(bn_2_4)
c_2_4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_4)

concat_2_3 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4])

bn_2_5  = BatchNormalization()(concat_2_3)
act_2_5 = Activation("relu")(bn_2_5)
c_2_5 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_2_5)

concat_2_4 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5])

bn_2_6  = BatchNormalization()(concat_2_4)
act_2_6 = Activation("relu")(bn_2_6)
c_2_6 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_6)

concat_2_5 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6])

bn_2_7 = BatchNormalization()(concat_2_5)
act_2_7 = Activation("relu")(bn_2_7)
c_2_7 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_2_7)

concat_2_6 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7])

bn_2_8 = BatchNormalization()(concat_2_6)
act_2_8 = Activation("relu")(bn_2_8)
c_2_8 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_8)

concat_2_7 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7, c_2_8])

bn_2_9  = BatchNormalization()(concat_2_7)
act_2_9 = Activation("relu")(bn_2_9)
c_2_9 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_2_9)

concat_2_8 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7, c_2_8, c_2_9])

bn_2_10  = BatchNormalization()(concat_2_8)
act_2_10 = Activation("relu")(bn_2_10)
c_2_10 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_10)

concat_2_9 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7, c_2_8, c_2_9, c_2_10])

bn_2_11  = BatchNormalization()(concat_2_9)
act_2_11 = Activation("relu")(bn_2_11)
c_2_11 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(act_2_11)

concat_2_10 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7, c_2_8, c_2_9, c_2_10, c_2_11])

bn_2_12  = BatchNormalization()(concat_2_10)
act_2_12 = Activation("relu")(bn_2_12)
c_2_12 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_12)

concat_2_11 = Concatenate()([c_2_1, c_2_2, c_2_3, c_2_4, c_2_5, c_2_6, c_2_7, c_2_8, c_2_9, c_2_10, c_2_11, c_2_12])

bn_2_13  = BatchNormalization()(concat_2_11)
act_2_13 = Activation("relu")(bn_2_13)
c_2_13 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_13)

concat_2_12 = Concatenate()([concat_2_11, c_2_13])

bn_2_14  = BatchNormalization()(concat_2_12)
act_2_14 = Activation("relu")(bn_2_14)
c_2_14 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_14)

concat_2_13 = Concatenate()([concat_2_12, c_2_14])

bn_2_15  = BatchNormalization()(concat_2_13)
act_2_15 = Activation("relu")(bn_2_15)
c_2_15 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_15)

concat_2_14 = Concatenate()([concat_2_13, c_2_15])

bn_2_16  = BatchNormalization()(concat_2_14)
act_2_16 = Activation("relu")(bn_2_16)
c_2_16 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_16)

concat_2_15 = Concatenate()([concat_2_14, c_2_16])

bn_2_17  = BatchNormalization()(concat_2_15)
act_2_17 = Activation("relu")(bn_2_17)
c_2_17 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_17)

concat_2_16 = Concatenate()([concat_2_15, c_2_17])

bn_2_18  = BatchNormalization()(concat_2_16)
act_2_18 = Activation("relu")(bn_2_18)
c_2_18 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_18)

concat_2_17 = Concatenate()([concat_2_16, c_2_18])

bn_2_19  = BatchNormalization()(concat_2_17)
act_2_19 = Activation("relu")(bn_2_19)
c_2_19 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_19)

concat_2_18 = Concatenate()([concat_2_17, c_2_19])

bn_2_20  = BatchNormalization()(concat_2_18)
act_2_20 = Activation("relu")(bn_2_20)
c_2_20 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_20)

concat_2_19 = Concatenate()([concat_2_18, c_2_20])

bn_2_20  = BatchNormalization()(concat_2_18)
act_2_20 = Activation("relu")(bn_2_20)
c_2_20 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_20)

concat_2_19 = Concatenate()([concat_2_18, c_2_20])

bn_2_21  = BatchNormalization()(concat_2_19)
act_2_21 = Activation("relu")(bn_2_21)
c_2_21 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_21)

concat_2_20 = Concatenate()([concat_2_19, c_2_21])

bn_2_22  = BatchNormalization()(concat_2_20)
act_2_22 = Activation("relu")(bn_2_22)
c_2_22 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_22)

concat_2_21 = Concatenate()([concat_2_20, c_2_22])

bn_2_23  = BatchNormalization()(concat_2_21)
act_2_23 = Activation("relu")(bn_2_23)
c_2_23 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_23)

concat_2_22 = Concatenate()([concat_2_21, c_2_23])


bn_2_24  = BatchNormalization()(concat_2_22)
act_2_24 = Activation("relu")(bn_2_24)
c_2_24 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_2_24)

concat_2_23 = Concatenate()([concat_2_22, c_2_24])

### dense 2

conv3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(concat_2_23)
avg_pool_2 = AveragePooling2D(pool_size=(2,2), strides=(2,2))(conv3)

# dense 3
bn_3_1 = BatchNormalization()(avg_pool_2)
act_3_1 = Activation("relu")(bn_3_1)
c_3_1 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_1)

bn_3_2 = BatchNormalization()(c_3_1)
act_3_2 = Activation("relu")(bn_3_2)
c_3_2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_3_2)

concat_3_1 = Concatenate()([c_3_1, c_3_2])

bn_3_3 = BatchNormalization()(concat_3_1)
act_3_3 = Activation("relu")(bn_3_3)
c_3_3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_3)

concat_3_2 = Concatenate()([c_3_1, c_3_2, c_3_3])

bn_3_4 = BatchNormalization()(concat_3_2)
act_3_4 = Activation("relu")(bn_3_4)
c_3_4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_3_4)

concat_3_3 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4])

bn_3_5 = BatchNormalization()(concat_3_3)
act_3_5 = Activation("relu")(bn_3_5)
c_3_5 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_5)

concat_3_4 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5])

bn_3_6 = BatchNormalization()(concat_3_4)
act_3_6 = Activation("relu")(bn_3_6)
c_3_6 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_6)

concat_3_5 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6])

bn_3_7 = BatchNormalization()(concat_3_5)
act_3_7 = Activation("relu")(bn_3_7)
c_3_7 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_7)

concat_3_6 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7])

bn_3_8 = BatchNormalization()(concat_3_6)
act_3_8 = Activation("relu")(bn_3_8)
c_3_8 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_8)

concat_3_7 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8])

bn_3_9 = BatchNormalization()(concat_3_7)
act_3_9 = Activation("relu")(bn_3_9)
c_3_9 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_9)

concat_3_8 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9])

bn_3_10 = BatchNormalization()(concat_3_8)
act_3_10 = Activation("relu")(bn_3_10)
c_3_10 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_10)

concat_3_9 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9, c_3_10])

bn_3_11 = BatchNormalization()(concat_3_9)
act_3_11 = Activation("relu")(bn_3_11)
c_3_11 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_11)

concat_3_10 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9, c_3_10, c_3_11])

bn_3_12 = BatchNormalization()(concat_3_10)
act_3_12 = Activation("relu")(bn_3_12)
c_3_12 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_12)

concat_3_11 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9, c_3_10, c_3_11, c_3_12])

bn_3_13 = BatchNormalization()(concat_3_11)
act_3_13 = Activation("relu")(bn_3_13)
c_3_13 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_13)

concat_3_12 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9, c_3_10, c_3_11, c_3_12, c_3_13])

bn_3_14 = BatchNormalization()(concat_3_12)
act_3_14 = Activation("relu")(bn_3_14)
c_3_14 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_14)

concat_3_13 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14])

bn_3_15 = BatchNormalization()(concat_3_13)
act_3_15 = Activation("relu")(bn_3_15)
c_3_15 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_15)

concat_3_14 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15])

bn_3_16 = BatchNormalization()(concat_3_14)
act_3_16 = Activation("relu")(bn_3_16)
c_3_16 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_16)

concat_3_15 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16])

bn_3_17 = BatchNormalization()(concat_3_15)
act_3_17 = Activation("relu")(bn_3_17)
c_3_17 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_17)

concat_3_16 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17])


bn_3_18 = BatchNormalization()(concat_3_16)
act_3_18 = Activation("relu")(bn_3_18)
c_3_18 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_18)

concat_3_17 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18])


bn_3_19 = BatchNormalization()(concat_3_17)
act_3_19 = Activation("relu")(bn_3_19)
c_3_19 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_19)

concat_3_18 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19])

bn_3_20 = BatchNormalization()(concat_3_18)
act_3_20 = Activation("relu")(bn_3_20)
c_3_20 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_20)

concat_3_19 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19, c_3_20])


bn_3_21 = BatchNormalization()(concat_3_19)
act_3_21 = Activation("relu")(bn_3_21)
c_3_21 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_20)

concat_3_20 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19, c_3_20,\
              c_3_21])


bn_3_22 = BatchNormalization()(concat_3_20)
act_3_22 = Activation("relu")(bn_3_22)
c_3_22 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_22)

concat_3_21 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19, c_3_20,\
              c_3_21, c_3_22])

bn_3_23 = BatchNormalization()(concat_3_21)
act_3_23 = Activation("relu")(bn_3_23)
c_3_23 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_23)

concat_3_22 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19, c_3_20,\
              c_3_21, c_3_22, c_3_23])

bn_3_24 = BatchNormalization()(concat_3_22)
act_3_24 = Activation("relu")(bn_3_24)
c_3_24 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_24)

concat_3_23 = Concatenate()([c_3_1, c_3_2, c_3_3, c_3_4, c_3_5, c_3_6, c_3_7, c_3_8, c_3_9,\
              c_3_10, c_3_11, c_3_12, c_3_13, c_3_14, c_3_15, c_3_16, c_3_17, c_3_18, c_3_19, c_3_20,\
              c_3_21, c_3_22, c_3_23, c_3_24])

bn_3_25 = BatchNormalization()(concat_3_23)
act_3_25 = Activation("relu")(bn_3_25)
c_3_25 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_25)

concat_3_24 = Concatenate()([concat_3_23, c_3_25])

bn_3_26 = BatchNormalization()(concat_3_24)
act_3_26 = Activation("relu")(bn_3_26)
c_3_26 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_26)

concat_3_25 = Concatenate()([concat_3_24, c_3_26])

bn_3_27 = BatchNormalization()(concat_3_25)
act_3_27 = Activation("relu")(bn_3_27)
c_3_27 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_27)

concat_3_26 = Concatenate()([concat_3_25, c_3_27])

bn_3_28 = BatchNormalization()(concat_3_26)
act_3_28 = Activation("relu")(bn_3_28)
c_3_28 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_28)

concat_3_27 = Concatenate()([concat_3_26, c_3_28])

bn_3_29 = BatchNormalization()(concat_3_27)
act_3_29 = Activation("relu")(bn_3_29)
c_3_29 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_3_29)

concat_3_28 = Concatenate()([concat_3_26, c_3_28])

# dense 3 ends

conv4 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same")(concat_3_23)
avg_pool_3 = AveragePooling2D(pool_size=(2,2), strides=(2,2))(conv4)

# dense 4 starts
bn_4_1 = BatchNormalization()(avg_pool_3)
act_4_1 = Activation("relu")(bn_4_1)
c_4_1 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_1)

bn_4_2 = BatchNormalization()(c_4_1)
act_4_2 = Activation("relu")(bn_4_2)
c_4_2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_4_2)

concat_4_1 = Concatenate()([c_4_1, c_4_2])

bn_4_3 = BatchNormalization()(concat_4_1)
act_4_3 = Activation("relu")(bn_4_3)
c_4_3 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_3)

concat_4_2 = Concatenate()([c_4_1, c_4_2, c_4_3])

bn_4_4 = BatchNormalization()(concat_4_2)
act_4_4 = Activation("relu")(bn_4_4)
c_4_4 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(act_4_4)

concat_4_3 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4])

bn_4_5 = BatchNormalization()(concat_4_3)
act_4_5 = Activation("relu")(bn_4_5)
c_4_5 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_5)

concat_4_4 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5])

bn_4_6 = BatchNormalization()(concat_4_4)
act_4_6 = Activation("relu")(bn_4_6)
c_4_6 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_6)

concat_4_5 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6])

bn_4_7 = BatchNormalization()(concat_4_5)
act_4_7 = Activation("relu")(bn_4_7)
c_4_7 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_7)

concat_4_6 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7])

bn_4_8 = BatchNormalization()(concat_4_6)
act_4_8 = Activation("relu")(bn_4_8)
c_4_8 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_8)

concat_4_7 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8])

bn_4_9 = BatchNormalization()(concat_4_7)
act_4_9 = Activation("relu")(bn_4_9)
c_4_9 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_9)

concat_4_8 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9])

bn_4_10 = BatchNormalization()(concat_4_8)
act_4_10 = Activation("relu")(bn_4_10)
c_4_10 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_10)

concat_4_9 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9, c_4_10])

bn_4_11 = BatchNormalization()(concat_4_9)
act_4_11 = Activation("relu")(bn_4_11)
c_4_11 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_11)

concat_4_10 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9, c_4_10, c_4_11])

bn_4_12 = BatchNormalization()(concat_4_10)
act_4_12 = Activation("relu")(bn_4_12)
c_4_12 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_12)

concat_4_11 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9, c_4_10, c_4_11, c_4_12])

bn_4_13 = BatchNormalization()(concat_4_11)
act_4_13 = Activation("relu")(bn_4_13)
c_4_13 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_13)

concat_4_12 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9, c_4_10, c_4_11, c_4_12, c_4_13])

bn_4_14 = BatchNormalization()(concat_4_12)
act_4_14 = Activation("relu")(bn_4_14)
c_4_14 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_14)

concat_4_13 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9,\
              c_4_10, c_4_11, c_4_12, c_4_13, c_4_14])

bn_4_15 = BatchNormalization()(concat_4_13)
act_4_15 = Activation("relu")(bn_4_15)
c_4_15 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_15)

concat_4_14 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9,\
              c_4_10, c_4_11, c_4_12, c_4_13, c_4_14, c_4_15])

bn_4_16 = BatchNormalization()(concat_4_14)
act_4_16 = Activation("relu")(bn_4_16)
c_4_16 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')(act_4_16)

concat_4_15 = Concatenate()([c_4_1, c_4_2, c_4_3, c_4_4, c_4_5, c_4_6, c_4_7, c_4_8, c_4_9,\
              c_4_10, c_4_11, c_4_12, c_4_13, c_4_14, c_4_15, c_4_16])


avg_pool_4 = AveragePooling2D(pool_size=(7,7))(concat_4_15)
dense_1 = Dense(1000)(avg_pool_4)
output = Activation("softmax")(dense_1)

model = Model(input=ip, output=output)
model.summary()