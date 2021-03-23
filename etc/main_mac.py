# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import cv2
import time
import numpy as np 
import pandas as pd 
import glob
import tensorflow as tf
# tf.enable_eager_execution()
from PIL import Image
# import tensorflow.keras as keras
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import convolve

plt.style.use('dark_background')

# %% [markdown]
# ## Data Loading & Preprocessing
# ___
# > 1. *'./data'* 에 있는 모든 '*.bmp'을 **os.listdir()** 을 이용해 접근 <br>
# > 2. **OCTAs**라는 list를 만들어 해당 파일들을 cv2.imread->numpy array로 전환<br>
# > 3. 읽어온 OCTAs 전체에 대해서 **normalizing**<br>
# >> OCTAs 의 모든 octa 데이터의 max를 출력했을 때 255 여서 255로 나눠줌<br>
# >> **"min-max normalizing"**

# %%
DATADIR = './data/og'
CROPDIR   = './data/crop'
files = os.listdir(DATADIR)
OCTAs = []
idx = 1
for f in files:
    if f.split('.')[-1]=='bmp':
        img = cv2.imread(os.path.join(DATADIR,f), cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        if idx==1:
            IMG_ROW, IMG_COL = np.shape(img)
            idx=0
        ####  여기서 크롭!!! ####
        c = (IMG_ROW-256)//2
        # img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        img = np.copy(img[c:IMG_ROW-c,c:IMG_COL-c])
        cv2.imwrite(os.path.join(CROPDIR,f), img)
        # Normalizing
        OCTAs.append(np.asarray(img/img.max().astype(float)))
    else: pass
IMG_ROW = IMG_ROW-c*2
IMG_COL = IMG_COL-c*2
# Normalizing
# OCTAs = np.divide(OCTAs, np.max(OCTAs))

# checking
print(np.shape(OCTAs))
plt.figure(figsize=(12,6))
plt.subplot(121),plt.hist(OCTAs[10])
plt.subplot(122),plt.imshow(OCTAs[10], cmap='gray')
plt.show()

# %% [markdown]
# ### Making New Data which make feel like comes from another device.
# > Adapt noise whole the OCTAs data so that we can feel its different from og.
# >> 1. Gaussian Noise : std=0.09 when image intensity (0~1) <br>
# >> 2. Frangi 2D : default option <br>
# >> 3. OCTA-Spectralis : og_img$**$1.5 <br>
# >> &nbsp&nbsp&nbsp  OCTA-Ciruss : ( gaussian\_noised\_img - 5000$*$frangi(gaussian\_noised\_img) )$**$1.5 , 

# %%
def make_noise(std, gray): 
    height, width = gray.shape 
    img_noise = np.zeros((height, width), dtype=np.float) 
    for i in range(height): 
        for a in range(width): 
            make_noise = np.random.normal() # 랜덤함수를 이용하여 노이즈 적용 
            set_noise = std * make_noise 
            img_noise[i][a] = gray[i][a] + set_noise 
    return img_noise


# %%
def show2OCTA(idx):
    o = OCTAs[idx]
    s = np.asarray(cv2.imread(os.path.join(SPECDIR,'{}'.format(files[idx])), cv2.IMREAD_GRAYSCALE))
    c = np.asarray(cv2.imread(os.path.join(CIRRDIR,'{}'.format(files[idx])), cv2.IMREAD_GRAYSCALE))

    plt.figure(figsize=(28, 14))
    plt.subplot(131), plt.imshow(o, cmap='gray')
    plt.title('OCTA-original',fontsize=15)
    plt.axis(False)

    plt.subplot(132), plt.imshow(s, cmap='gray')
    plt.title('OCTA-Spectralis',fontsize=15)
    plt.axis(False)
    
    plt.subplot(133), plt.imshow(c, cmap='gray')
    plt.title('OCTA-Cirrus',fontsize=15)
    plt.axis(False)
    plt.show()


# %%
def contrast_en(img,upper,lower):
    lowerpercentile, upperpercentile = lower, upper

    if lowerpercentile is not None:
        qlow = np.percentile(img, lowerpercentile)
    if upperpercentile is not None:
        qup = np.percentile(img, upperpercentile)

    if lowerpercentile is not None:
        img[img < qlow] = qlow
    if upperpercentile is not None:
        img[img > qup] = qup
    return img


# %%
import shutil

'''
from preprocessing.frangiFilter2D import FrangiFilter2D

# Make Spectralis Directory and Cirrus Directory.
# And Put Images which have been preprocessed.
try :
    SPECDIR = os.path.join('.\\data', 'spectralis')
    CIRRDIR = os.path.join('.\\data', 'cirrus')
    os.mkdir(SPECDIR)
    os.mkdir(CIRRDIR)
    std = 0.09
    for idx, octa in enumerate(OCTAs):
        gauss_noise = make_noise(std, octa)
        
        spectralis  = (gauss_noise-5500*FrangiFilter2D(gauss_noise)[0])**1.4
        cirrus = contrast_en(np.copy(octa), 100,2)
        
        cv2.imwrite(os.path.join(CIRRDIR,f'{files[idx]}'), cirrus*255)
        cv2.imwrite(os.path.join(SPECDIR,f'{files[idx]}'), spectralis*255)
except:
    pass


# %%
# idx = 43
# show2OCTA(idx)

# %%
SPECs = os.listdir(SPECDIR)
CIRRs = os.listdir(CIRRDIR)

s_octa = []
c_octa = []
for spec in SPECs:
    s_img=np.asarray((cv2.imread(os.path.join(SPECDIR,spec), cv2.IMREAD_GRAYSCALE)))
    c_img=np.asarray((cv2.imread(os.path.join(CIRRDIR,spec), cv2.IMREAD_GRAYSCALE)))
    s_img = s_img/s_img.max().astype(float)
    c_img = c_img/c_img.max().astype(float)
    s_octa.append(s_img)
    c_octa.append(c_img)

# print(np.shape(s_octa))
# print(np.shape(c_octa))

# %% [markdown]
# ### Split the data - train, valid, test 
# ___
# > In paper, they use 43 training sets, 15 validation sets, 7 test sets

# %%
from sklearn.model_selection import train_test_split

indices = [idx for idx in range(len(SPECs))]
tot_train_idx, test_idx = train_test_split(indices,       test_size=7)
train_idx,    valid_idx = train_test_split(tot_train_idx, test_size=15)

train_sp = np.asarray([s_octa[a] for a in train_idx])
train_cr = np.asarray([c_octa[a] for a in train_idx])

valid_sp = np.asarray([s_octa[b] for b in valid_idx])
valid_cr = np.asarray([s_octa[b] for b in valid_idx])

test_sp  = np.asarray([s_octa[c] for c in test_idx])
test_cr  = np.asarray([c_octa[c] for c in test_idx])


train_sp = train_sp.reshape(-1, IMG_ROW, IMG_COL, 1)
train_cr = train_cr.reshape(-1, IMG_ROW, IMG_COL, 1)
valid_sp = valid_sp.reshape(-1, IMG_ROW, IMG_COL, 1)
valid_cr = valid_cr.reshape(-1, IMG_ROW, IMG_COL, 1)
test_sp  = test_sp.reshape(-1, IMG_ROW, IMG_COL, 1)
test_cr  = test_cr.reshape(-1, IMG_ROW, IMG_COL, 1)

print('train_sp :',np.shape(train_sp),'!!')
print('train_cr :',np.shape(train_cr),'!!')

# # plot the splitted data 
# plt.figure(figsize=(12,12))
# slice_idx = np.random.randint(len(train_sp))
# plt.subplot(121), plt.imshow(train_sp[slice_idx], cmap='gray'),plt.axis(False),plt.title(f'Spectralis train[{slice_idx}]', fontsize=20)
# plt.subplot(122), plt.imshow(train_cr[slice_idx], cmap='gray'),plt.axis(False),plt.title(f'Cirrus train[{slice_idx}]', fontsize=20)
# plt.subplots_adjust(wspace=0)
# plt.show()
'''

#%%
from sklearn.model_selection import train_test_split

# '''
CROPs = os.listdir(CROPDIR)
crop_octa = []

for og in CROPs:
    img = np.asarray(cv2.imread(os.path.join(CROPDIR,og),cv2.IMREAD_GRAYSCALE))
    img = img/img.max().astype(float)
    crop_octa.append(img)


indices = [idx for idx in range(len(CROPs))]
tot_train_idx, test_idx = train_test_split(indices,       test_size=7)
train_idx,    valid_idx = train_test_split(tot_train_idx, test_size=15)

train_crop = (np.asarray([crop_octa[a] for a in train_idx])).reshape(-1, IMG_ROW, IMG_COL, 1)
valid_crop = (np.asarray([crop_octa[b] for b in valid_idx])).reshape(-1, IMG_ROW, IMG_COL, 1)
test_crop  = (np.asarray([crop_octa[c] for c in test_idx])).reshape(-1, IMG_ROW, IMG_COL, 1)


# train_crop = train_sp.reshape(-1, IMG_ROW, IMG_COL, 1)
# valid_crop = valid_sp.reshape(-1, IMG_ROW, IMG_COL, 1)
# test_crop  = test_sp.reshape(-1, IMG_ROW, IMG_COL, 1)

print('train_sp :',np.shape(train_crop),'!!')
# '''


#%%
img_shape = (IMG_ROW, IMG_COL, 1)    
batch_size = 4
latent_dim = 2

# %%
from SEGNET.segnet import segnet
# from SEGNET.synnet import synnet
from keras.utils import plot_model
from keras.optimizers import Adam

model_seg = segnet(input_shape=img_shape, output_mode='sigmoid', pool_size=2)
# model_syn = synnet(input_shape=img_shape, output_mode='sigmoid', pool_size=2)
# model.summary()

model_seg.compile(loss='mse', optimizer = Adam(lr=0.001), metrics=['accuracy'])

# model_syn.compile(loss='mse', 
#               optimizer = Adam(lr=0.00005),
#               metrics=['accuracy'])



# plot_model(model, show_shapes=True, to_file='.\\segnet_model.png')
# print("input :",train_crop.shape)
# %%
# ------------------- call back functions -------------------
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
EPOCH = 999
checkpoint_path = './checkpoint/epoch_'+str(EPOCH)+'.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='loss', 
                             verbose=1)

early_stopping = EarlyStopping(monitor='loss', patience=50)

# Define the Keras TensorBoard callback.
from datetime import datetime

logdir="./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


#%%
# '''
hist = model_seg.fit(train_crop, train_crop, 
                batch_size=10, 
                epochs=EPOCH, 
                verbose = 2,
                callbacks=[checkpoint, tensorboard_callback],
                validation_data = [valid_crop,valid_crop]
            )
# %%

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y'    , label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper left')
plt.savefig('./data/resul/learning_curve_e{}.png'.format(EPOCH))
plt.show()
# '''
# %%

def edge_processing(image_):
    img_i      = (image_*255).astype(np.uint8)
    edge_i     = cv2.Canny(img_i, 250,255)
    img_f      = (img_i/255).astype(float)
    edge_f     = (edge_i/255).astype(float)
    diff_i     = (img_f - edge_f**0.01)*255+edge_i**0.94
    _,binary_i = cv2.threshold(diff_i,255//2.5, 255,cv2.THRESH_BINARY)

    return binary_i



# %%
model_seg.load_weights(checkpoint_path)
print(model_seg.output)
test_imgs = model_seg.predict(test_crop)
# binary thresholding
# test_b = np.where(test_imgs > .5, 1.0, 0.0).astype('float32')

for idx, d in enumerate(test_imgs):
    test_out = d.reshape(IMG_ROW, IMG_COL)
    _, thresh1 = cv2.threshold(test_out,0.5, 1,cv2.THRESH_BINARY)
    edge_filtered = edge_processing(test_out)

    # cv2.imwrite(f'.\\data\\result\\test_b_{test_idx[idx]}_e{EPOCH}.png',thresh1*255)
    # cv2.imwrite(f'.\\data\\result\\test_{test_idx[idx]}_e{EPOCH}.png'  ,d*255)
    # cv2.imwrite(f'.\\data\\result\\og_{test_idx[idx]}_e{EPOCH}.png'    ,crop_octa[test_idx[idx]].reshape(IMG_ROW, IMG_COL)*255)    

    plt.subplot(121), plt.imshow(crop_octa[test_idx[idx]].reshape(IMG_ROW, IMG_COL)*255,cmap='gray'), plt.title('testimage_input_{}_img'.format(test_idx[idx]), fontsize=15)
    plt.subplot(122), plt.imshow(test_out*255,cmap='gray')      , plt.title('testimage_output_{}_img'.format(test_idx[idx]), fontsize=15)
    # plt.subplot(223), plt.imshow(thresh1 ,cmap='gray'), plt.title(f'binary_img_{test_idx[idx]}_img', fontsize=15)
    # plt.subplot(224), plt.imshow(edge_filtered,cmap='gray'), plt.title(f'edge_filtered_img_{test_idx[idx]}_img', fontsize=15)
    plt.show()
# %%
from sampling import sampling

# plt.imshow(sampling(test_imgs)[3], cmap='gray')
# plt.show()

seg_out = sampling(test_imgs)

# %%
'''
from tensorflow.keras import Sequential
from SEGNET.vicce import vicce

model_vic = vicce(input_shape=img_shape, output_mode='sigmoid', pool_size=2)
model_vic.compile(loss='mse', 
              optimizer = Adam(lr=[0.001, 0.00005]),
              metrics=['accuracy'])
model_vic.fit(  [train_sp, train_cr],[train_cr, train_sp],
                batch_size=10, 
                epochs=EPOCH, 
                verbose = 2,
                callbacks=[checkpoint, tensorboard_callback],
                validation_data = [valid_crop,valid_crop]
                )
# model = Sequential()
# model.add(segnet)

# train_ =(train_sp, train_cr)
# valid_ =valid_sp, valid_cr
# test_ =test_sp, test_cr
# def vicce(train_, valid_, test_):
#     # ps = segnet(train_[0], train_[0])
#     # ps_ = sampling(ps)
#     # c_hat = synnet(ps_, train_[1])

#     # pc = segnet(train_[1], train_[1])
#     # pc_ = sampling(pc)
#     # s_hat = synnet(pc_, train_[0])

#     model_seg = segnet(input_shape=img_shape, output_mode='sigmoid', pool_size=2)
#     model_syn = synnet(input_shape=img_shape, output_mode='sigmoid', pool_size=2)
#     # model.summary()

#     model_seg.compile(loss='mse', 
#                 optimizer = Adam(lr=0.001),
#                 metrics=['accuracy'])

#     model_syn.compile(loss='mse', 
#                 optimizer = Adam(lr=0.00005),
#                 metrics=['accuracy'])


#     return 
# %%
'''