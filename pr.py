import os
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
import tensorflow_probability as tfp

CROPDIR = "./data/crop"
CROPs = os.listdir(CROPDIR)
crop_octa = []

for og in CROPs:
    img = np.asarray(cv2.imread(os.path.join(CROPDIR,og),cv2.IMREAD_GRAYSCALE))
    xmax, xmin = img.max(), img.min()
    img = (img - xmin)/(xmax - xmin) # min, max Normalizing : 0~1
    img = img.astype(float)
    img = img*2-1 # modify the range to [-1~1]
    crop_octa.append(img)

crop_octa = np.asarray(crop_octa)
datas = tf.reshape(crop_octa,(65,256,256,1))
datas.numpy().shape

train = datas[:43]
valid = datas[43:-7]
test = datas[-7:]

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

plt.figure(figsize=(20,100))
for idx in range(1,7,2):
    plt.subplot(7,2,idx), plt.imshow(np.reshape(test[idx], (256,256)), cmap="gray"), plt.axis(False), plt.title('input image')
    plt.subplots_adjust(wspace=0, hspace=0)
    a = sigmoid(np.reshape(test[idx], (256,256)))
    plt.subplot(7,2,idx), plt.imshow(a, cmap="gray"), plt.axis(False), plt.title('input image')
    plt.subplots_adjust(wspace=0, hspace=0)

    cv2.imwrite('sigout.png', a)
    # #predict for SegNet
    # segnet_outs = SegNet.predict(tf.reshape(test[idx],(1,256,256,1)))
    # reconstructed = segnet_outs.reshape(256, 256)
    
    # plt.subplot(7,2,idx+1), plt.imshow(reconstructed, cmap="gray"), plt.axis(False), plt.title('output image')
    # plt.subplots_adjust(wspace=0, hspace=0.1)

plt.show()

