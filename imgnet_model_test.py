
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

def pre_Model(part):
	myunet = myUnet()
	model = myunet.get_unet()
	model.load_weights(part + '_dice_4_unet_3.hdf5',by_name = False)
	model.save(part + '_model.h5')
	
	print('model save done!')
        
def dice_coef(y_true, y_pred):
	smooth = 1.0
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
		
def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

# In[3]:

if __name__ == '__main__':
	if sys.argv[1] == 'liver' or sys.argv[1] == 'spleen':
		pre_Model(sys.argv[1])
		imgs = os.listdir('./' + sys.argv[1] +"/img")
		print("load model")
		model = load_model(sys.argv[1] + '_model.h5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
		data = np.empty((1,512,512,1))
		img0 = cv2.imread('./' + sys.argv[1] + '/img/' + imgs[1])
		img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
		data[0,:,:,0] = img / 255
		imgs_mask_test = model.predict(data, batch_size=1, verbose=1)
		cv2.imwrite('tmp0.jpg',imgs_mask_test[0,:,:,0] * 255)
		print('model predict done!')
	else:
		print('the argv is not corrected!')
#     img = cv2.imread('tmp.jpg')
#     print(img.shape)
#     img = img * 255
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show() 
    

