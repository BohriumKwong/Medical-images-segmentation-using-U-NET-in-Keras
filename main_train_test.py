
# coding: utf-8

# In[1]:


import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import gc

def load_Img(imgDir,imgFoldName,subFold):
    imgs = os.listdir(imgDir+"/"+imgFoldName)
    labels = os.listdir(imgDir+"/"+subFold)
    imgNum = len(imgs)
    labelNum = len(labels)
    #input size is 512 * 512 * 1

    data = np.empty((imgNum * 2,512,512,1))
    label = np.empty((imgNum * 2,512,512,1))
    if labelNum == imgNum:
        print (imgDir + ' imgNum: ',imgNum)
        for i in range (imgNum):
            img = cv2.imread(imgDir+"/"+imgFoldName+"/"+imgs[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img0 = img / 255            
            arr_0 = cv2.flip(img0,1)         
            
            data[i,:,:,0] = img0
            data[i + imgNum,:,:,0] = arr_0         

            img = cv2.imread(imgDir+"/"+ subFold +"/"+labels[i])
            img0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            arr_0 = cv2.flip(img0,1)                            
            label[i,:,:,0] = img0
            label[i + imgNum,:,:,0] = arr_0
            gc.collect()
            
    return data,label


# In[6]:

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization,concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.callbacks import TensorBoard
from keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth ) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return - dice_coef(y_true, y_pred)


def get_unet():
        inputs = Input((512, 512, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)


        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
        print('model compile')
#         model.summary()
        return model
#     binary_crossentropy
    
def get_callbacks(filepath,part,patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    tb_log = TensorBoard(log_dir='./log/' + part)
    return [es, msave,tb_log]


# In[3]:

if __name__ == '__main__':
    liver_data, liver_label =load_Img('liver','img','cls')
    print (liver_data.shape)
#     spleen_data, spleen_label =load_Img('spleen','img','cls')
#     print (spleen_data.shape)


# In[4]:

x_train,x_test,y_train,y_test = train_test_split(liver_data, liver_label, random_state=0, train_size=0.4)
#     del liver_data
#     del liver_label
#     gc.collect()


# In[ ]:

def data_generator(data, targets, batch_size):       
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    while True:
        for i in range(int(data.shape[0]/batch_size+1)):
            yield data[i * batch_size:(i + 1) * batch_size], targets[i * batch_size:(i + 1) * batch_size]    
model = get_unet()
model.load_weights('liver2_8_unet.hdf5',by_name = False)
file_path = "liver_dice_8_unet.hdf5"
callbacks_s = get_callbacks(file_path,'liver',patience=5)
batch_size = 256
model.fit_generator(data_generator(x_train, y_train,batch_size), steps_per_epoch = 4128 * 0.6//8,epochs=7, verbose=1,
        validation_data = (x_test, y_test),validation_steps =4128 * 0.4//8 ,callbacks = callbacks_s)
with open('liver_dice_8_gen.txt','w') as f:
    f.write(str(model.history))

     batch_size = 128
     with tf.device('/gpu:0'):
		model.fit_generator(data_generator( x_train, y_train,batch_size), steps_per_epoch = (y_train.shape[0] + batch_size - 1) // batch_size,
                             nb_epoch=2,verbose=1,callbacks = callbacks_s,validation_data = data_generator( x_test, y_test,batch_size))
#########################################################################################################


# In[ ]:

model = get_unet()
model.load_weights('liver2_8_unet.hdf5',by_name = False)
file_path = "liver_dice_8_unet.hdf5"
callbacks_s = get_callbacks(file_path,'liver', patience=5)
model.fit(liver_data,liver_label, batch_size=8, epochs=7, verbose=1,validation_split=0.3, shuffle=True,callbacks = callbacks_s)
with open('liver_dice_8.txt','w') as f:
    f.write(str(model.history))


# In[ ]:

del liver_data
del liver_label

gc.collect()
spleen_data, spleen_label =load_Img('spleen','img','cls')
model2 = get_unet()
model2.load_weights('spleen2_8_unet.hdf5',by_name = False)
file_path = "spleen_dice_8_unet.hdf5"
callbacks_s = get_callbacks(file_path,'spleen', patience=5)
model2.fit(spleen_data,spleen_label, batch_size=8, epochs=7, verbose=1,validation_split=0.3, shuffle=True,callbacks = callbacks_s)



# In[ ]:

json_string = model2.to_json()
with open('./liver_model_architecture.json','w') as json_file:
    json_file.write(json_string)

