# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob, os

########################################################################################
# # dcm = dicom.read_file('x55_liver.dcm')
# image = sitk.ReadImage('./nii/422_l.nii')
# # image2 = sitk.ReadImage('x55_liver.dcm')
# # dcm2 = dicom.read_file('./python2/SYN00274')
# # dcm.LoadPrivateTagsOn()
# # print(dcm.ReadImageInformation())
# print('GetDirection     :',image.GetDirection())
# # print('x55_liver.dcm:',image2.GetDirection())
# print('GetOrigin     :',image.GetOrigin())
# # print('x55_liver.dcm:',image2.GetOrigin())
# # print(image)
# # image2 = mudicom.load('/python2/SYN00274')
# image_array = sitk.GetArrayFromImage(image).astype(np.int16)[0,:,:]
#
#
# sp = image_array.shape
# print(sp)
#
# Img = np.zeros([sp[0],sp[1],3])
# Img2 =Img
# # Img[:,:,0]=img
# # Img[:,:,1]=img
# # Img[:,:,2]=img
# #
# Img2[:,:,0]=image_array
# Img2[:,:,1]=image_array
# Img2[:,:,2]=image_array
#
# imgs = os.listdir('../图像分割/CHESS1701/湘雅/肝中介/423')
# print(type(imgs))
# print(imgs)
# for i in range(len(imgs)):
#     if imgs[i].find('nii')>=0:
#         print(imgs[i] + ' is a nii file!')
#     else:
#         print(imgs[i] + ' is not a nii file!')

# Img2 = Img2 * 255
# cv2.imshow("img",Img2)
# cv2.waitKey(0)
# Img[:,:,0]=image[0,:,:,]
# Img[:,:,1]=image[0,:,:,]
# Img[:,:,2]=image[0,:,:,]
# cv2.imshow(Img)
# cv2.imwrite('SYN00274_dcm.jpg',Img)
# cv2.imwrite('./nii/422_l.jpg',Img2)
###################################################################################
def load_Img(imgDir):
    miss_record = ''
    #根目录，如IMG
    dir_1 = os.listdir('./CHESS1701/'+imgDir)
    for i in range(len(dir_1)):
        #根目录下的医院目录，如世纪坛
        dir_2 = os.listdir('./CHESS1701/'+imgDir+'/'+dir_1[i])
        for j in range(len(dir_2)):
            #医院目录下的部位目录，如世纪坛/liver
            dir_3 = os.listdir('./CHESS1701/'+imgDir+'/'+dir_1[i]+'/'+dir_2[j])
            for k in range(len(dir_3)):
                #部位目录下的子目录，如世纪坛/liver/271
                dir_4 = os.listdir('./CHESS1701/'+imgDir+'/'+dir_1[i]+'/'+dir_2[j]+'/'+dir_3[k])
                count_nill  = 0
                for l in range(len(dir_4)):
                    if dir_4[l].find('nii')>=0:
                        count_nill = count_nill + 1
                    #检查子目录下的nii图片数量
                if count_nill<2:
                    ####
                    miss_record = miss_record + dir_1[i]+'/'+dir_2[j]+'/'+dir_3[k] +'原始图像或标记图像缺失,只有' + str(count_nill) + '张！' +'\r\n'
#                else:
#                    for l in range(len(dir_4)):
#                        if dir_4[l].find('nii')>=0:
#                            #遍历所有文件，将名字包含nii的文件读入,保存为3通道
#                            image = sitk.ReadImage('./CHESS1701/'+imgDir+'/'+dir_1[i]+'/'+dir_2[j]+'/'+dir_3[k]+'/'+dir_4[l])
#                            image_array = sitk.GetArrayFromImage(image).astype(np.int16)[0,:,:]
#                            img_res = np.zeros([image_array.shape[0],image_array.shape[1],3])
#                            img_res[:,:,0]=image_array
#                            img_res[:,:,1]=image_array
#                            img_res[:,:,2]=image_array
#                            if dir_4[l].find('_')>=0:
#                                cv2.imwrite('./CHESS1701/'+dir_2[j]+'/cls'+'/'+dir_1[i]+'_'+dir_3[k]+'.jpg',img_res)
#                            else:
#                                cv2.imwrite('./CHESS1701/'+dir_2[j]+'/img'+'/'+dir_1[i]+'_'+dir_3[k]+'.jpg',img_res)
                        #print(dir_1[i]+'/'+dir_2[j]+'/'+dir_3[k]+'/'+' 完成!')
                    print (dir_1[i]+'/'+dir_2[j]+'/'+dir_3[k]+'count_nill: '+ str(count_nill))

        print('医院目录： '+dir_1[i]+ '全部执行完毕!')
    with open('./CHESS1701/' + 'miss.txt', 'w') as txt_file:
             txt_file.write(miss_record)

if __name__ == '__main__':
    import sys
    load_Img('IMG')
    # image_z = sitk.ReadImage('./nii/bio/422_l.nii')
    # dir = os.listdir('H:/生物圖騰/图像分割/CHESS1701/IMG/世纪坛/liver/271/')
    # print(os.listdir('H:/生物圖騰/图像分割/CHESS1701/IMG/世纪坛/liver/271/'))
    # for x in range(len(dir)):
    #     print('H:/生物圖騰/图像分割/CHESS1701/IMG/世纪坛/liver/271/'+dir[x])
        # os.system('cd '+ 'H:/生物圖騰/图像分割/CHESS1701/IMG/世纪坛/liver/271')
        # image_z = sitk.ReadImage(dir[x])
