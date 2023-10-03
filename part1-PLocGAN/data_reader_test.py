from torch.utils.data.dataset import Dataset
from config import *
from common_util import *
from tool import *
import cv2

class data_reader(Dataset):
    def __init__(self):

        if not ope(TrainLabels_h5):
            raise (FileNotFoundError("The file {} is not found!".format(TrainLabels_h5)))
        if not ope(ValLabels_h5):
            raise (FileNotFoundError("The file {} is not found!".format(ValLabels_h5)))
        if not ope(TrainImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(TrainImgs_basedir)))
        if not ope(ValImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(ValImgs_basedir)))

        self.imgbasedir = TestImgs_basedir
        self.labels = self.read_h5(TestLabels_h5)

    def read_h5(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        data = h5file.get('labels').value
        return data

    def read_imgs(self,basedir,idx):
        BGimg = cv2.imread(opj(basedir,'BGImg'+str(idx)+'.bmp'))
        img = np.zeros([256,256,3],dtype='float32')
        img[:, :, 0] = BGimg[:, :, 0]
        img[:, :, 1] = BGimg[:, :, 1]
        img[:, :, 2] = BGimg[:, :, 2]
        img = img/255
        return img

    def find_imgs(self, index):
        curimg = self.read_imgs(self.imgbasedir,index)
        curlabel = self.labels[index].astype('float32')
        curimg = image_to_tensor(curimg)
        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return len(self.labels)