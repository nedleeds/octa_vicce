# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import Libraries which gonna be needed

# %%
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from tensorflow         import keras
from tensorflow.keras   import layers
from keras.models       import Model
from keras.layers.core  import Activation, Reshape
from keras.layers       import MaxPool2D, UpSampling2D, concatenate, Dropout, Dense, Flatten, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

# %% [markdown]
# ## Loading DataSet

# %%
CROPDIR = "/Share/data/crop"
CROPs = os.listdir(CROPDIR)
crop_octa = []

for og in CROPs:
    img = np.asarray(cv2.imread(os.path.join(CROPDIR,og),cv2.IMREAD_GRAYSCALE))
    img = img/img.max().astype(float)
    crop_octa.append(img)
crop_octa = np.asarray(crop_octa)
datas = tf.reshape(crop_octa,(65,256,256,1))
datas.numpy().shape

train = datas[:43]
valid = datas[43:-7]
test = datas[-7:]

# %% [markdown]
# 
# ## Data Check
# ___
# * 1. Check whether the normalized data is correct<br>
#  -> Should be between 0.0 ~ 1.0
# 
# 
#  
# * 2. Split the normalized data to Train, Valid, Test <br>
#  -> Train : 43/65, Valid : 15/65, Test : 7/65

# %%
plt.figure(figsize=(15,7))
plt.subplot(121),plt.hist(crop_octa[0].reshape((256,256))), plt.title('crop_octa', fontsize=15)
plt.subplot(122),plt.hist(datas[0].numpy().reshape((256,256))), plt.title('data', fontsize=15)
plt.show()

print('train tensors :', train.shape)
print('valid tensors :', valid.shape)
print('test tensors :', test.shape)

# %% [markdown]
# ## SegNet

# %%
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.utils import multi_gpu_model
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding   = padding
        self.pool_size = pool_size
        self.strides   = strides

    def call(self, inputs, **kwargs):
        padding   = self.padding
        pool_size = self.pool_size
        strides   = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1] # ksize = [ 1, 2, 2, 1 ] 
            padding = padding.upper() # "same", "valid" , ... ==> "SAME", "VALID"
            strides = [1, strides[0], strides[1], 1] # strides[0], strdies[1] ==> [1, 2, 2 ,1]
            print(inputs.shape)
            # inputs = tf.expand_dims(inputs, axis=0, name=None)
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding,
                include_batch_in_index=True
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = tf.cast(argmax, tf.dtypes.float32)
        return [output, argmax]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]  # inputs : pool_#, mask(has indices info)
        mask = tf.cast(mask, "int32")
        input_shape = tf.shape(updates)
        #  calculation new shape
        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3],
            )
        self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype="int32")
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf .size(updates)
        # updates_size = updates.shape
        indices = K.transpose(tf.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values  = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

latent_dim = 2
input_shape = (256, 256, 1)
pool_size = (2,2)
kernel = 3
stride = 1
output_mode = "sigmoid"

# Level-1
inputs = Input(shape=input_shape)
print(inputs.shape)
conv_1 = Convolution2D(8, (kernel,kernel), padding="same", data_format='channels_last')(inputs)
conv_1 = BatchNormalization()(conv_1)
conv_1 = Activation("relu")(conv_1)
conv_2 = Convolution2D(8, (kernel,kernel), padding="same", data_format='channels_last')(conv_1)
conv_2 = BatchNormalization()(conv_2)
conv_2 = Activation("relu")(conv_2)      #512x512x8   -> concate 1
conv_2 = Dropout(0.5)(conv_2, training=True)

pool_1 = MaxPooling2D(pool_size, data_format='channels_last')(conv_2)
# pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)


# Level-2
conv_3 = Convolution2D(16, (kernel,kernel), padding="same", data_format='channels_last')(pool_1)
conv_3 = BatchNormalization()(conv_3)
conv_3 = Activation("relu")(conv_3)
conv_4 = Convolution2D(16, (kernel,kernel), padding="same", data_format='channels_last')(conv_3)
conv_4 = BatchNormalization()(conv_4)
conv_4 = Activation("relu")(conv_4)     #256x256x16 # -> concate 2
conv_4 = Dropout(0.5)(conv_4, training=True)

pool_2 = MaxPooling2D(pool_size, data_format='channels_last')(conv_4) 
# pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)


# Level-3
conv_5 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(pool_2)
conv_5 = BatchNormalization()(conv_5)
conv_5 = Activation("relu")(conv_5)
conv_6 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(conv_5)
conv_6 = BatchNormalization()(conv_6)
conv_6 = Activation("relu")(conv_6)
conv_7 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(conv_6)
conv_7 = BatchNormalization()(conv_7)
conv_7 = Activation("relu")(conv_7)     #128x128x32 # -> concate 3
conv_7 = Dropout(0.5)(conv_7, training=True)

pool_3 = MaxPooling2D(pool_size, data_format='channels_last')(conv_7) 
# pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)


# Level-4
conv_8 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(pool_3)
conv_8 = BatchNormalization()(conv_8)
conv_8 = Activation("relu")(conv_8)
conv_9 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(conv_8)
conv_9 = BatchNormalization()(conv_9)
conv_9 = Activation("relu")(conv_9)
conv_10 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(conv_9)
conv_10 = BatchNormalization()(conv_10)
conv_10 = Activation("relu")(conv_10)   #64x64x64  -> concate 4 
conv_10 = Dropout(0.5)(conv_10, training=True)

pool_4 = MaxPooling2D(pool_size, data_format='channels_last')(conv_10) 
# pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)


# Level-5
conv_11 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(pool_4)
conv_11 = BatchNormalization()(conv_11)
conv_11 = Activation("relu")(conv_11)
conv_12 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(conv_11)
conv_12 = BatchNormalization()(conv_12)
conv_12 = Activation("relu")(conv_12)
conv_13 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(conv_12)
conv_13 = BatchNormalization()(conv_13)
conv_13 = Activation("relu")(conv_13)     
conv_13 = Dropout(0.5)(conv_13, training=True)

pool_5 = MaxPooling2D(pool_size, data_format='channels_last')(conv_13)   
# pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)


print("Build enceder done..")


# decoder
# Level-5 : Lowest level -> don't use skip connection.
unpool_1 = UpSampling2D(interpolation='bilinear')(pool_5)
# unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
# print(unpool_1)

conv_14 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(unpool_1)
conv_14 = BatchNormalization()(conv_14)
conv_14 = Activation("relu")(conv_14)
conv_15 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(conv_14)
conv_15 = BatchNormalization()(conv_15)
conv_15 = Activation("relu")(conv_15)
conv_16 = Convolution2D(128, (kernel,kernel), padding="same", data_format='channels_last')(conv_15)
conv_16 = BatchNormalization()(conv_16)
conv_16 = Activation("relu")(conv_16) 
conv_16 = Dropout(0.5)(conv_16, training=True)

# Level-4
unpool_2 = UpSampling2D(interpolation='bilinear', data_format='channels_last')(conv_16) 
# unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])
# skip connection : conv_10(64x64x64) + unpool_2(64x62x128)
unpool_2 = concatenate([unpool_2, conv_10],axis=-1)

conv_17 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(unpool_2) #64x64x64
conv_17 = BatchNormalization()(conv_17)
conv_17 = Activation("relu")(conv_17)
conv_18 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(conv_17)
conv_18 = BatchNormalization()(conv_18)
conv_18 = Activation("relu")(conv_18)
conv_19 = Convolution2D(64, (kernel,kernel), padding="same", data_format='channels_last')(conv_18)
conv_19 = BatchNormalization()(conv_19)
conv_19 = Activation("relu")(conv_19) #64x64x64
conv_19 = Dropout(0.5)(conv_19, training=True)

# Level-3
unpool_3 = UpSampling2D(interpolation='bilinear', data_format='channels_last')(conv_19) #128x128x64
# unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])
# skip connection with conv_7(128x128x32) + unpool_3(128x128x64)
unpool_3 = concatenate([unpool_3, conv_7],axis=-1)

conv_20 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(unpool_3)
conv_20 = BatchNormalization()(conv_20)
conv_20 = Activation("relu")(conv_20)
conv_21 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(conv_20)
conv_21 = BatchNormalization()(conv_21)
conv_21 = Activation("relu")(conv_21)
conv_22 = Convolution2D(32, (kernel,kernel), padding="same", data_format='channels_last')(conv_21)
conv_22 = BatchNormalization()(conv_22)
conv_22 = Activation("relu")(conv_22) #128x128x32
conv_22 = Dropout(0.5)(conv_22, training=True)

# Level-2
unpool_4 = UpSampling2D(interpolation='bilinear', data_format='channels_last')(conv_22) #256x256x32
# unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])
# skip connection with conv_2(256x256x16) + unpool_4(256x256x32)
unpool_4 = concatenate([unpool_4, conv_4],axis=-1)

conv_23 = Convolution2D(16, (kernel), padding="same", data_format='channels_last')(unpool_4)
conv_23 = BatchNormalization()(conv_23)
conv_23 = Activation("relu")(conv_23)
conv_24 = Convolution2D(16, (kernel), padding="same", data_format='channels_last')(conv_23)
conv_24 = BatchNormalization()(conv_24)
conv_24 = Activation("relu")(conv_24)  
conv_24 = Dropout(0.5)(conv_24, training=True)

# Level-1
unpool_5 = UpSampling2D(interpolation='bilinear', data_format='channels_last')(conv_24) #512x512x16
# unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])
# skip connection with conv_1(512x512x8)+unpool_5(512x512x16)
unpool_5 = concatenate([unpool_5, conv_2],axis=-1)

conv_25 = Convolution2D(8, (kernel), padding="same", data_format='channels_last')(unpool_5) #512x512x8
conv_25 = BatchNormalization()(conv_25)
conv_25 = Activation("relu")(conv_25)
conv_26 = Convolution2D(1, (kernel, kernel), padding="same", data_format='channels_last')(conv_25) #512x512x1
conv_26 = BatchNormalization()(conv_26)
conv_26 = Reshape(input_shape)(conv_26)

outputs = Activation(output_mode)(conv_26) #sigmoid

print("Build decoder done..")

SegNet = Model(inputs=inputs, outputs=outputs, name="SegNet")
SegNet.summary()

# %% [markdown]
# ## Callback for SegNet

# %%
import shutil

CURRDIR = os.getcwd()
CHCKDIR = os.path.join(CURRDIR, 'checkpoint')
LOGSDIR = os.path.join(CURRDIR, 'logs')

if os.path.isdir(CHCKDIR) :
    shutil.rmtree(CHCKDIR)
if os.path.isdir(LOGSDIR):
    shutil.rmtree(LOGSDIR)
elif not os.path.isdir(CHCKDIR): 
    os.mkdir(CHCKDIR)
elif not os.path.isdir(LOGSDIR):
    os.mkdir(LOGSDIR)
else:
    pass
print("LOGSDIR = {}".format(LOGSDIR))
print("CHCKDIR = {}".format(CHCKDIR))
seg_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss' ), 
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHCKDIR,'model.{epoch:02d}-{loss:.2f}.h5'),
        save_best_only=True 
        ),
    tf.keras.callbacks.TensorBoard(log_dir = LOGSDIR)
]


# %%

SegNet.compile(loss='mse', optimizer = keras.optimizers.Adam(lr=0.00001), metrics=['accuracy'])

SegNet.fit(
            train, train,
            epochs=1500, batch_size= 4,
            callbacks=seg_callbacks,
            validation_data = [valid, valid]
            )


# %%
# # 손실과 옵티마이저
# loss_fn   = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# # 훈련을 위해 모델을 설정합니다.
# encoder_seg.compile(optimizer, loss=loss_fn)

# # 모델을 훈련합니다.
# decoder_seg.fit(dataset, epochs=1)


# %%
import matplotlib.pyplot as plt
plt.style.use('dark_background')

plt.figure(figsize=(20,100))
for idx in range(1,7,2):
    plt.subplot(7,2,idx), plt.imshow(np.reshape(test[idx], (256,256)), cmap="gray"), plt.axis(False), plt.title('input image')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    #predict for SegNet
    segnet_outs = SegNet.predict(tf.reshape(test[idx],(1,256,256,1)))
    reconstructed = segnet_outs.reshape(256, 256)
    
    plt.subplot(7,2,idx+1), plt.imshow(reconstructed, cmap="gray"), plt.axis(False), plt.title('output image')
    plt.subplots_adjust(wspace=0, hspace=0.1)

plt.show()


# %%
#


# %%
#


# %%
#


# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



