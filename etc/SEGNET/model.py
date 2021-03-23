# from keras.layers import Input, Dropout
# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Activation, Reshape
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model
from SEGNET.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,Dense,Flatten,Dropout,BatchNormalization, Reshape, ReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np

def segnet():
    # encoder
    inputs = tf.keras.Input(shape=(512,512,1))
    kernel = 3
    pool_size=2
    drop_rate = 0.5

    conv_1 = Conv2D(8, (kernel,kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = ReLU(conv_1)
    conv_2 = Conv2D(8, (kernel,kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = ReLU(conv_2)
    conv_2 = Dropout(drop_rate)(conv_2)
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(16, (kernel,kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = ReLU(conv_3)
    conv_4 = Conv2D(16, (kernel,kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = ReLU(conv_4)
    conv_4 = Dropout(drop_rate)(conv_4)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(32, (kernel,kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = ReLU(conv_5)
    conv_6 = Conv2D(32, (kernel,kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = ReLU(conv_6)
    conv_7 = Conv2D(32, (kernel,kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = ReLU(conv_7)
    conv_7 = Dropout(drop_rate)(conv_7)
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(64, (kernel,kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = ReLU(conv_8)
    conv_9 = Conv2D(64, (kernel,kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = ReLU(conv_9)
    conv_10 = Conv2D(64, (kernel,kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = ReLU(conv_10)
    conv_10 = Dropout(drop_rate)(conv_10)
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(128, (kernel,kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = ReLU(conv_11)
    conv_12 = Conv2D(128, (kernel,kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = ReLU(conv_12)
    conv_13 = Conv2D(128, (kernel,kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = ReLU(conv_13)
    conv_13 = Dropout(drop_rate)(conv_13)
    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
    # dropout = Dropout(rate=drop_rate)(unpool_1) 
    conv_14 = Conv2DTranspose(128, (kernel,kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = ReLU(conv_14)
    conv_15 = Conv2DTranspose(128, (kernel,kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = ReLU(conv_15)
    conv_16 = Conv2DTranspose(128, (kernel,kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = ReLU(conv_16)
    conv_16 = Dropout(drop_rate)(conv_16)
    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Conv2DTranspose(64, (kernel,kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = ReLU(conv_17)
    conv_18 = Conv2DTranspose(64, (kernel,kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = ReLU(conv_18)
    conv_19 = Conv2DTranspose(64, (kernel,kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = ReLU(conv_19)
    conv_19 = Dropout(drop_rate)(conv_19)
    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2DTranspose(32, (kernel,kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = ReLU(conv_20)
    conv_21 = Conv2DTranspose(32, (kernel,kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = ReLU(conv_21)
    conv_22 = Conv2DTranspose(32, (kernel,kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = ReLU(conv_22)
    conv_22 = Dropout(drop_rate)(conv_22)
    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2DTranspose(16, (kernel,kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = ReLU(conv_23)
    conv_24 = Conv2DTranspose(16, (kernel,kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = ReLU(conv_24)
    conv_24 = Dropout(drop_rate)(conv_24)
    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2DTranspose(8, (kernel,kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = ReLU(conv_25)

    conv_26 = Conv2DTranspose((1, 1), padding="same")(conv_25)
#     conv_26 = BatchNormalization()(conv_26)
    
#     conv_26 = Reshape(
#         (n_labels, input_shape[1]*input_shape[1]),
#         input_shape=(n_labels, input_shape[0], input_shape[1]),
#     )(conv_26)

    # outputs = Activation(output_mode)(conv_26)
    outputs = conv_26
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model
