from torch.utils.data.dataset import Dataset
from data_process.data_preprocess import data_preprocess
from config.config_ import *
from utils.common_util import *
from datasets.tool import *
import cv2
import pickle
import pandas as pd
from utils.augment_util_ import train_vit_transform

class data_reader(Dataset):
    def __init__(self,net,Images_Norm_params,is_Train=True,is_Syn=True,transform=None):

        if not ope(TrainLabels_Classfication_h5):
            raise (FileNotFoundError("The file {} is not found!".format(TrainLabels_Classfication_h5)))
        if not ope(ValLabels_Classfication_h5):
            raise (FileNotFoundError("The file {} is not found!".format(ValLabels_Classfication_h5)))
        if not ope(TrainImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(TrainImgs_basedir)))
        if not ope(ValImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(ValImgs_basedir)))

        self.Images_Norm_params = Images_Norm_params
        self.is_Train = is_Train
        self.is_Syn = is_Syn
        self.transform = transform
        self.net = net
        if is_Train:
            self.imgbasedir = TrainImgs_basedir
            self.labels = self.read_h5_classfication(TrainLabels_Classfication_h5)
        else:
            if is_Syn:
                self.imgbasedir = ValSynImgs_basedir
                self.labels = self.read_h5_classfication(ValSynLabels_Classfication_h5)
            else:
                self.imgbasedir = ValImgs_basedir
                self.labels = self.read_h5_classfication(ValLabels_Classfication_h5)
        self.num = len(self.labels)

    def read_h5_classfication(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        data = h5file['labels'][()]
        return data

    def read_imgs_classfication(self,basedir,idx,label=None):
        BGYimg = cv2.imread(opj(basedir,'BGYImg'+str(idx)+'.png'))
        if not dataset == 'lymo':
            img = np.zeros([256,256,3],dtype='float32')
            img[:, :, 0] = BGYimg[:, :, 0]
            img[:, :, 1] = BGYimg[:, :, 1]
            img[:, :, 2] = BGYimg[:, :, 2]
        else:
            img = np.zeros([256, 256, 2], dtype='float32')
            img[:, :, 0] = BGYimg[:, :, 0]
            img[:, :, 1] = BGYimg[:, :, 1]
        curlabel = self.labels[idx].astype('float32')
        idx_1 = np.where(curlabel == 1)
        if self.net == 'ViT':
            img = train_vit_transform(img)

        if self.transform is not None:
            p = [0.55,0.15,0.15,0.15]
            if len(curlabel[idx_1]) == 2:
                if 'gan' in self.net.lower():
                    img,curlabel,flag = self.transform(img/255,self.labels[idx].astype('float32'),p)
                    return img,curlabel.astype('float32'),flag
                else:
                    img = self.transform(img,p)
                    img = img/255
                    return img
            else:
                if 'gan' in self.net.lower():
                    return img/255, curlabel.astype('float32'), 0
                else:
                    img = img / 255
                    return img
        else:
            img = img / 255
            return img


    def changelabel(self, label):
        idx = np.where(label == 1)
        if len(label[idx]) == 2:
            comb = self.label_5to2(label)
            if comb == 'None':
                raise ValueError('comb can not be None!')
            idx = np.array(idx).squeeze()
            chantmplabel = [0.5,0.5]
            for n in range(2):
                label[idx[n]] = chantmplabel[n]
        return label

    def changelabel_real(self, label):
        idx = np.where(label>0)
        if len(label[idx]) == 2:
            idx = np.array(idx).squeeze()
            chantmplabel = [0.5,0.5]
            for n in range(2):
                label[idx[n]] = chantmplabel[n]
        return label

    def label_5to2(self,label):
        label_rev_anno = {
            0: 'Cy', 1: 'Np',2: 'Mi',3: 'Nu',
            4: 'PM',5: 'CyNp',6: 'CyPM',7: 'NuMi',8: 'NuNu'
        }
        comb = 'None'
        binlabel = label > 0
        for i in baselabel.keys():
            if (baselabel[i] == binlabel).all():
                comb = label_rev_anno[i]
                break
        return comb

    def find_imgs(self, index):
        flag = 0
        if 'gan' in self.net.lower():
            if self.transform is not None:
                curimg,curlabel,flag = self.read_imgs_classfication(self.imgbasedir,index)
            else:
                curimg = self.read_imgs_classfication(self.imgbasedir, index)
                curlabel = self.labels[index].astype('float32')
        else:
            curimg = self.read_imgs_classfication(self.imgbasedir,index)
            curlabel = self.labels[index].astype('float32')

        curimg = self.extract_imgs(curimg)
        if not self.is_Syn:
            if 'gan' in self.net.lower():
                if not flag:
                    if not dataset == 'lymo':
                        curlabel = self.changelabel(curlabel)
                    else:
                        curlabel =self.changelabel_real(curlabel)
            else:
                if not dataset == 'lymo':
                    curlabel = self.changelabel(curlabel)
                else:
                    curlabel = self.changelabel_real(curlabel)
        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def extract_imgs(self,imgs):
        for i in range(self.Images_Norm_params['in_channels']):
            mean = np.squeeze(self.Images_Norm_params['mean'])
            std = np.squeeze(self.Images_Norm_params['std'])
            imgs[:,:,i] = (imgs[:,:,i] - mean[i]) / std[i]
        images = image_to_tensor(imgs)
        return images

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return self.num