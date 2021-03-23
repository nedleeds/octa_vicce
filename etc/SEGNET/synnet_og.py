from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from SEGNET.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.layers import MaxPooling2D, UpSampling2D, concatenate, Dropout
import keras.layers as layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def synnet(input_shape=(256,256,1), n_labels=1, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    # encoder
    # Level-1
    fineset_ch = 16
    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(fineset_ch, (kernel,kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(fineset_ch, (kernel,kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)      #512x512x8   -> concate 1
    conv_2 = Dropout(0.5)(conv_2, training=True)
    # pool_1 = MaxPooling2D(pool_size)(conv_2)
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    # Level-2
    conv_3 = Convolution2D(fineset_ch*2, (kernel,kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(fineset_ch*2, (kernel,kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)     #256x256x16 # -> concate 2
    conv_4 = Dropout(0.5)(conv_4)
    # pool_2 = MaxPooling2D(pool_size)(conv_4) 
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    # Level-3
    conv_5 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)     #128x128x32 # -> concate 3
    conv_7 = Dropout(0.5)(conv_7, training=True)
    # pool_3 = MaxPooling2D(pool_size)(conv_7) 
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    # Level-4
    conv_8 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)   #64x64x64  -> concate 4 
    conv_10 = Dropout(0.5)(conv_10, training=True)
    # pool_4 = MaxPooling2D(pool_size)(conv_10) 
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    # Level-5
    conv_11 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)     
    conv_13 = Dropout(0.5)(conv_13, training=True)
    # pool_5 = MaxPooling2D(pool_size)(conv_13)   
    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    
    print("Build enceder done..")
    

    # decoder
    # Level-5 : Lowest level -> don't use skip connection.
    # unpool_1 = UpSampling2D(interpolation='bilinear')(pool_5) 
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
    conv_14 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(fineset_ch*16, (kernel,kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16) 
    conv_16 = Dropout(0.5)(conv_16, training=True)
    unpool_2 = UpSampling2D(interpolation='bilinear')(conv_16) 

    # Level-4
    # skip connection : conv_10(64x64x64) + unpool_2(64x62x128)
    # unpool_2 = concatenate([unpool_2, conv_10],axis=-1)
    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
    conv_17 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(unpool_2) #64x64x64
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(fineset_ch*8, (kernel,kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19) #64x64x64
    conv_19 = Dropout(0.5)(conv_19, training=True)
    unpool_3 = UpSampling2D(interpolation='bilinear')(conv_19) #128x128x64

    # Level-3
    # skip connection with conv_7(128x128x32) + unpool_3(128x128x64)
    # unpool_3 = concatenate([unpool_3, conv_7],axis=-1)
    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])
    conv_20 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(fineset_ch*4, (kernel,kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22) #128x128x32
    conv_22 = Dropout(0.5)(conv_22, training=True)
    unpool_4 = UpSampling2D(interpolation='bilinear')(conv_22) #256x256x32

    # Level-2
    # skip connection with conv_2(256x256x16) + unpool_4(256x256x32)
    # unpool_4 = concatenate([unpool_4, conv_4],axis=-1)
    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])
    conv_23 = Convolution2D(fineset_ch*2, (kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(fineset_ch*2, (kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)  
    conv_24 = Dropout(0.5)(conv_24, training=True)
    unpool_5 = UpSampling2D(interpolation='bilinear')(conv_24) #512x512x16
    # unpool_5 = 256x256 

    # Level-1
    # skip connection with conv_1(512x512x8)+unpool_5(512x512x16)
    # unpool_5 = concatenate([unpool_5, conv_2],axis=-1)
    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])
    conv_25 = Convolution2D(fineset_ch, (kernel), padding="same")(unpool_5) #512x512x8
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)
    conv_26 = Convolution2D(1, (1, 1), padding="valid")(conv_25) #512x512x1
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(input_shape)(conv_26)

    outputs = Activation(output_mode)(conv_26) #sigmoid 
    tf.summary.image('output_image', tf.reshape(outputs,[-1,28,28,1]))
   
   
   
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SynNet") 

    return model 