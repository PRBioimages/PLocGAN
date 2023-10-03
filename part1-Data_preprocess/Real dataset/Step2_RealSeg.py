# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from scipy.spatial import Voronoi
import cv2
import numpy as np
import os
import mahotas as mh
import csv
from skimage import filters, morphology
import shutil
import cv2
import math
from scipy.io import loadmat
# from restoration import rolling_ball

def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')

from scipy import ndimage

def scfilter(image, iterations, kernel):
    """
    Sine‐cosine filter.
    kernel can be tuple or single value.
    Returns filtered image.
    """
    for n in range(iterations):
        image = np.arctan2(
        ndimage.filters.uniform_filter(np.sin(image), size=kernel),
        ndimage.filters.uniform_filter(np.cos(image), size=kernel))
    return image

def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def SegImageVoronoi(ImagePath,scale):
    # Use a breakpoint in the code line below to debug your script.
    Zero_Flag = 0
    gray = cv2.imread(ImagePath,0)  # 图片读取
    gray = gray * scale
    dnaf = mh.gaussian_filter(gray, 3).astype('uint8')
    # plt.imshow(dnaf)
    # plt.show()
    if cv2.countNonZero(dnaf) == 0:
        Zero_Flag = 1
        nuclei = cv2.cvtColor(dnaf, cv2.COLOR_GRAY2BGR)
        return Zero_Flag, nuclei, nuclei, 0
    # plt.imshow(dnaf)
    # plt.show()

    # plt.imshow(dnaf)
    # plt.show()
    thresh = filters.threshold_otsu(dnaf)
    binary = morphology.opening(dnaf > thresh, morphology.disk(15))
    labeled, nr_nuclei = mh.label(binary)
    print(nr_nuclei)
    # plt.imshow(labeled)
    # plt.show()
    seeds, nr_nuclei = mh.labeled.filter_labeled(labeled, remove_bordering=False, min_size=3000)

    nuclei = cv2.cvtColor(dnaf, cv2.COLOR_GRAY2BGR)
    if nr_nuclei < 1:
        Zero_Flag = 1
        return Zero_Flag, nuclei, nuclei, nr_nuclei
    else:
        SupplementMask= mh.segmentation.gvoronoi(seeds)
        # plt.imshow(SupplementMask)
        # plt.show()
        return Zero_Flag, SupplementMask, nuclei, nr_nuclei

def preprocess():
    base_dir = 'F:\GAN_data\RealData2010'
    sec_dir = 'gnf_images_MitoLyso2\images\MitoLyso'
    store_base_dir = os.path.join(base_dir,'Preprocessed_',sec_dir)
    train_dir = os.path.join(base_dir,sec_dir)
    mkdir(store_base_dir)
    for tmpFile in os.listdir(train_dir):
        print(tmpFile)
        if 'readme' in tmpFile.lower():
            shutil.copy(os.path.join(train_dir, tmpFile),store_base_dir)
            continue
        else:
            store_img_dir = os.path.join(store_base_dir, tmpFile)
            mkdir(store_img_dir)
            meangrays = []
            for tmpimg in os.listdir(os.path.join(train_dir, tmpFile)):
                if 'factors' in tmpimg.lower():
                    continue
                else:
                    # intensity_scale = loadmat(os.path.join(train_dir, tmpFile,'factors.mat'))['factors']
                    fir = tmpimg.split(".bmp")[0]
                    sec = int(fir.split('-')[1])
                    thir = fir[len(fir) - 1]
                    if int(thir) % 2 == 0:
                        ImagePath = os.path.join(train_dir, tmpFile, tmpimg)
                        gray = cv2.imread(ImagePath, 0)
                        # gray = gray*intensity_scale[0,sec-1]
                        dft = np.fft.fft2(gray)
                        dft_shift = np.fft.fftshift(dft)
                        fftgray = np.log(1+np.abs(dft_shift))
                        fftgray_ = fftgray[fftgray.shape[0]//2-30:fftgray.shape[0]//2+29,fftgray.shape[1]//2-30:fftgray.shape[1]//2+29]
                        tmpImg_Mean = np.mean(fftgray_)
                        meangrays.append(tmpImg_Mean)
            # plt.hist(meangrays)
            # plt.show()
            a = np.nonzero(meangrays)[0]
            threshold = np.sum(meangrays)/len(np.nonzero(meangrays)[0])#-np.std(np.array(meangrays)[a])#np.mean(green_meangrays) np.sum(Y)/len(np.nonzero(Y)[0])
            print(threshold)

            for tmpimg in os.listdir(os.path.join(train_dir,tmpFile)):
                if 'factors' in tmpimg.lower():
                    shutil.copy(os.path.join(train_dir, tmpFile,tmpimg), store_img_dir)
                    continue
                else:
                    # intensity_scale = loadmat(os.path.join(train_dir, tmpFile, 'factors.mat'))['factors']
                    fir = tmpimg.split(".bmp")[0]
                    sec = int(fir.split('-')[1])
                    thir = fir[len(fir) - 1]
                    if int(thir) % 2 == 0:
                        ImagePath = os.path.join(train_dir, tmpFile, tmpimg)
                        gray = cv2.imread(ImagePath, 0)
                        # gray = gray * intensity_scale[0,sec - 1]
                        dft = np.fft.fft2(gray)
                        dft_shift = np.fft.fftshift(dft)
                        fftgray = np.log(1+np.abs(dft_shift))
                        fftgray_ = fftgray[fftgray.shape[0] // 2 - 30:fftgray.shape[0] // 2 + 29,
                                   fftgray.shape[1] // 2 - 30:fftgray.shape[1] // 2 + 29]
                        if np.mean(fftgray_) >= threshold:
                            print(tmpimg)
                            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            # gray = clahe.apply(gray)
                            rgbimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(os.path.join(store_img_dir, tmpimg), rgbimg)
                            protein = cv2.imread(os.path.join(train_dir, tmpFile, 'Stack-%05d.bmp' % (sec - 1)), 0)
                            # protein = protein * intensity_scale[0,sec - 2]
                            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            # protein = clahe.apply(protein)
                            protein = cv2.cvtColor(protein, cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(os.path.join(store_img_dir,'Stack-%05d.bmp' % (sec - 1)), protein)



# Press the green button in the gutter to run the script.
def cal_mask_area(mask_base_dir, tmpFile):
    area = []
    for tmpmask in os.listdir(os.path.join(mask_base_dir, tmpFile)):
        if 'factors' in tmpmask.lower():
            continue
        else:
            fir = tmpmask.split(".bmp")[0]
            thir = fir[len(fir) - 1]
            if int(thir) % 2 == 0:
                continue
            else:
                MaskPath = os.path.join(mask_base_dir, tmpFile, tmpmask)
                Mask = cv2.imread(MaskPath, 0)>0
                area.append(np.sum(Mask))
    area = np.array(area)
    lower_q = np.quantile(area, 0.75)
    std = 0
    mean = 0
    thre = lower_q
    return std, mean,thre

def mulprotein(train_dir,mask_base_dir,tmpFile,tmpmask,store_img_dir):
    MaskPath = os.path.join(mask_base_dir, tmpFile, tmpmask)
    tmpimg = tmpmask.replace('StackMask', 'Stack')
    ImagePath = os.path.join(train_dir, tmpFile, tmpimg)
    Img = cv2.imread(ImagePath, 0)
    Mask = cv2.imread(MaskPath, 0)
    ImgByMask = Img * (Mask > 0)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ImgByMask = clahe.apply(ImgByMask)
    ImgByMask = cv2.cvtColor(ImgByMask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(store_img_dir, tmpimg), ImgByMask)

def muldna(train_dir,tmpFile,tmpmask,store_img_dir):
    tmpimg = tmpmask.replace('StackMask', 'Stack')
    ImagePath = os.path.join(train_dir, tmpFile, tmpimg)
    Img = cv2.imread(ImagePath, 0)
    ImgByMask = Img
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ImgByMask = clahe.apply(ImgByMask)
    ImgByMask = cv2.cvtColor(ImgByMask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(store_img_dir, tmpimg), ImgByMask)


def mul_mask():
    # preprocess()
    base_dir = 'F:\GAN_data\RealData2010'
    sec_dir = 'gnf_images_MitoLyso2\images\MitoLyso'
    store_base_dir = os.path.join(base_dir, 'Preprocessed_La', sec_dir)
    mask_base_dir = os.path.join(base_dir, 'MLMask', sec_dir)
    train_dir = os.path.join(base_dir, sec_dir)
    mkdir(store_base_dir)
    for tmpFile in os.listdir(mask_base_dir):
        print(tmpFile)
        store_img_dir = os.path.join(store_base_dir, tmpFile)
        mkdir(store_img_dir)
        _,_,low_value = cal_mask_area(mask_base_dir, tmpFile)
        for tmpmask in os.listdir(os.path.join(mask_base_dir, tmpFile)):
            if 'factors' in tmpmask.lower():
                continue
            else:
                fir = tmpmask.split(".bmp")[0]
                sec = int(fir.split('-')[1])
                thir = fir[len(fir) - 1]
                if int(thir) % 2 == 0:
                    continue
                else:
                    print(tmpmask)
                    MaskPath = os.path.join(mask_base_dir, tmpFile, tmpmask)
                    Mask = cv2.imread(MaskPath, 0)>0
                    # if np.sum(Mask) >= low_value:
                    mulprotein(train_dir,mask_base_dir, tmpFile, tmpmask, store_img_dir)
                    tmpmaskdna = 'StackMask-%05d.bmp' % (sec + 1)
                    mulprotein(train_dir,mask_base_dir, tmpFile, tmpmaskdna, store_img_dir)

if __name__ == '__main__':
    # preprocess()
    mul_mask()
    base_dir = 'F:\GAN_data\RealData2010'
    sec_dir = 'gnf_images_MitoLyso2\images\MitoLyso'
    store_base_dir = os.path.join(base_dir,'SingleCell_La',sec_dir)
    train_dir = os.path.join(base_dir,'Preprocessed_La',sec_dir)
    mkdir(store_base_dir)
    for tmpFile in os.listdir(train_dir):
        print(tmpFile)
        if 'readme' in tmpFile.lower():
            shutil.copy(os.path.join(train_dir, tmpFile),store_base_dir)
            continue
        else:
            store_img_dir = os.path.join(store_base_dir, tmpFile)
            mkdir(store_img_dir)
            CSV_dir = os.path.join(store_img_dir, 'ZeroCellName.csv')
            FalseCSV_dir = os.path.join(store_img_dir, 'MultiCellName.csv')
            f = open(CSV_dir, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Name'])
            Falsef = open(FalseCSV_dir, 'w', encoding='utf-8', newline='')
            Falsecsv_writer = csv.writer(Falsef)
            Falsecsv_writer.writerow(['Name','newName', 'Seeds'])
            Mask_dir = os.path.join(store_img_dir, 'StepAMask')
            print(Mask_dir)
            FalsenewSingleCell_dir = os.path.join(store_img_dir, 'StepARawdata')  # FinalData
            mkdir(Mask_dir)
            mkdir(FalsenewSingleCell_dir)

            n = 0
            for tmpimg in os.listdir(os.path.join(train_dir,tmpFile)):
                if 'factors' in tmpimg.lower():
                    shutil.copy(os.path.join(train_dir, tmpFile,tmpimg), store_img_dir)
                    continue
                else:
                    # intensity_scale = loadmat(os.path.join(train_dir, tmpFile, 'factors.mat'))['factors']
                    print(tmpimg)
                    fir = tmpimg.split(".bmp")[0]
                    sec = int(fir.split('-')[1])
                    thir = fir[len(fir) - 1]
                    ImagePath = os.path.join(train_dir, tmpFile, tmpimg)
                    if int(thir) % 2 == 0:
                        Single_Flag, SupplementMask, nuclei, nr_nuclei = SegImageVoronoi(ImagePath,1)  # Voronoi
                        if Single_Flag == 1:
                            csv_writer.writerow([ImagePath])  # real cell image
                            print('No Cell...')
                        else:
                            cv2.imwrite(FalsenewSingleCell_dir + '/BlueImag' + str(n)+'.bmp', nuclei)
                            cv2.imwrite(Mask_dir + '/Mask'  + str(n)+'.bmp', SupplementMask)
                            Falsecsv_writer.writerow([ImagePath,'BlueImag' + str(n) ,nr_nuclei])  # false cell image
                            print('Multi-Cell...')
                            protein = cv2.imread(os.path.join(train_dir, tmpFile, 'Stack-%05d.bmp' % (sec - 1)), 0)
                            # protein = protein * intensity_scale[0,sec - 2]
                            protein = cv2.cvtColor(protein, cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(FalsenewSingleCell_dir + '/GreenImag' + str(n) + '.bmp', protein)
                            n = n+1
            f.close()
            Falsef.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
