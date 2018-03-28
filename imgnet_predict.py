
# coding: utf-8

# In[1]:


import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from unet import *
from PIL import Image


# In[2]:

def pre_Img(imgDir):
    imgs = os.listdir(imgDir+"/img")
    labels = os.listdir(imgDir+"/cls")
    imgNum = len(imgs)
    labelNum = len(labels)
    data = np.empty((1,512,512,1))
    myunet = myUnet()
    model = myunet.get_unet()
    model.load_weights('spleen1_liver_8_unet.hdf5',by_name = False)
    if labelNum == imgNum:
        for i in range (imgNum // 50 ):
            img0 = cv2.imread(imgDir+"/img/"+imgs[i])
            label = cv2.imread(imgDir+"/cls/"+labels[i])
            label = label * 255
            img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
            label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
            data[0,:,:,0] = img / 255
            imgs_mask_test = model.predict(data, batch_size=1, verbose=1)
#             print (imgs_mask_test.shape)
#             imgs_mask_0 = Image.fromarray(imgs_mask_test[0,:,:,0].astype('int8') * 255)
            cv2.imwrite('tmp.jpg',imgs_mask_test[0,:,:,0])
            imgs_mask = cv2.imread('tmp.jpg',0)
            imgs_mask = imgs_mask * 255
            _, binary = cv2.threshold(imgs_mask,127,255,cv2.THRESH_BINARY)
            _, binary_l = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
            _, contours, _, = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            _, contours_l, _, = cv2.findContours(binary_l,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #红色是标记的，绿色是预测的
            cv2.drawContours(img0,contours,-1,(0,255,0),3)
            cv2.drawContours(img0,contours_l,-1,(0,0,255),3)
            cv2.imwrite('./result/' + imgDir + '/'+ imgs[i],img0)
    print('predict done!')
        


# In[3]:

if __name__ == '__main__':
    pre_Img('liver')
#     img = cv2.imread('tmp.jpg')
#     print(img.shape)
#     img = img * 255
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show() 
    

