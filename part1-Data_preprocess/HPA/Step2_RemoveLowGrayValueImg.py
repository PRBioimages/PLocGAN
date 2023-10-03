
import shutil
import os
import pandas as pd
import numpy as np
import cv2
def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')

# train_dir = 'D:/Program/Proj02CGANCellFusionData/HPA'
train_dir = 'D:/Program/HPA'
for tmpFile in os.listdir(train_dir):
    print(tmpFile)
    if tmpFile !='Nucleoli':
        continue
    else:
        copyold_dir = train_dir + '/' + tmpFile + '/SingleData'
        old_dir = train_dir + '/' + tmpFile + '/SingleData'
        new_dir = train_dir + '/' + tmpFile + '/StepDChosenSingleData_4'
        mkdir(new_dir)

        GreenImg_GrayHistMean = []
        GeneralGreenImg_GrayHistMean = 0
        for tmpImgMask in os.listdir(old_dir):
            # print(tmpImgMask)
            tmpmask_name = tmpImgMask.split("_segmentation")
            mask_name = tmpmask_name[0]
            if tmpImgMask.find('green.jpg') != -1:
                GreenImgPath = os.path.join(old_dir, tmpImgMask)
                GreenImg = cv2.imread(GreenImgPath)
                tmpGreenImg_Mean = np.mean(GreenImg)
                GreenImg_GrayHistMean.append(tmpGreenImg_Mean)

    print('It is time to select')
    GeneralGreenImg_GrayHistMean = np.mean(GreenImg_GrayHistMean)
    for tmpImgMask2 in os.listdir(old_dir):
        print(tmpImgMask2)
        tmpmask_name2 = tmpImgMask2.split("_segmentation")
        mask_name2 = tmpmask_name2[0]
        if tmpImgMask2.find('green.jpg') != -1:
            greenImgName = tmpImgMask2
            GreenImgPath2 = os.path.join(old_dir, tmpImgMask2)
            GreenImg2 = cv2.imread(GreenImgPath2)[:, :, 1]
            tmpGreenImg_GrayHistMean2 = np.mean(GreenImg2)
            if tmpGreenImg_GrayHistMean2 > GeneralGreenImg_GrayHistMean:
                imgpath = os.path.join(copyold_dir, greenImgName)
                newpath = os.path.join(new_dir, greenImgName)
                shutil.copy(imgpath, newpath)
                shutil.copy(
                    imgpath.replace(
                        'green.jpg', 'blue.jpg'), newpath.replace(
                        'green.jpg', 'blue.jpg'))
                shutil.copy(
                    imgpath.replace(
                        'green.jpg', 'yellow.jpg'), newpath.replace(
                        'green.jpg', 'yellow.jpg'))
                shutil.copy(
                    imgpath.replace(
                        'green.jpg', 'red.jpg'), newpath.replace(
                        'green.jpg', 'red.jpg'))
                oldMaskdir = train_dir + '/' + tmpFile + '/MaskSeg'
                newMaskdir = train_dir + '/' + tmpFile + '/StepDMaskchosen_4'
                mkdir(newMaskdir)
                try:
                    maskname = greenImgName.replace('_green.jpg', '.jpg')
                    oldMaskpath = os.path.join(oldMaskdir, maskname)
                    newMaskpath = os.path.join(newMaskdir, maskname)
                    shutil.copy(oldMaskpath, newMaskpath)
                except:
                    maskname = greenImgName.replace('_green.jpg', '.png')
                    oldMaskpath = os.path.join(oldMaskdir, maskname)
                    newMaskpath = os.path.join(newMaskdir, maskname)
                    shutil.copy(oldMaskpath, newMaskpath)
