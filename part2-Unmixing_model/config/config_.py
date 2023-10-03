import numpy as np

dataset = 'hpa'
if 'lymo' == dataset.lower():
  NUM_CLASSES   = 2
  CATE = 1
else:
  NUM_CLASSES   = 5
  CATE = 4
BATCH_SIZE = 32
COMB =0
MODE = 'densenet'

baselabel = {
    0: [1, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0],
    2: [0, 0, 1, 0, 0],
    3: [0, 0, 0, 1, 0],
    4: [0, 0, 0, 0, 1],
    5: [1, 1, 0, 0, 0],
    6: [1, 0, 0, 0, 1],
    7: [0, 1, 1, 0, 0],
    8: [0, 1, 0, 1, 0]
}
quantilabel = {
    0: np.array([[0.25,0.75]]).astype('float32'),
    1: np.array([[0.5,0.5]]).astype('float32'),
    2: np.array([[0.75,0.25]]).astype('float32')
}

comb_anno = {'CyNp': 0,'CyPM': 1,'NuMi': 2,'NuNu': 3}
label_anno = {'Cy': 0,'Np': 1,'Mi': 2,'Nu': 3,'PM': 4,'CyNp': 5,'CyPM': 6,'NuMi': 7,'NuNu': 8}

PRETRAINED_DIR = "../pretrained"
if 'lymo' == dataset.lower():
  TrainLabels_Classfication_h5 = '../rawdata/lymo2010/lymo2010Trainlabels.h5'
  ValLabels_Classfication_h5 = '../rawdata/lymo2010/lymo2010Vallabels.h5'
  TestLabels_h5 = '../rawdata/lymo2010/lymo2010Testlabels.h5'
  TestImgs_basedir = '../rawdata/lymo2010/test'
  ##
  TrainImgs_basedir = '../rawdata/lymo2010/train'
  ValImgs_basedir = '../rawdata/lymo2010/val'
  Imgs_mean_std = '../rawdata/lymo2010/TrainImgslymo_Mean_Std.mat'
else:
  TrainLabels_Classfication_h5 = '../rawdata/AllRCateTrainLabels_HPA.h5'
  ValSynLabels_Classfication_h5 = '../rawdata/ValSynLabels_HPA.h5'
  ValLabels_Classfication_h5 = '../rawdata/AllRCateValLabels_HPA.h5'
  ##
  TrainImgs_basedir = '../rawdata/HPA/TrainImgs'
  ValSynImgs_basedir = '../rawdata/HPA/ValSynImages'
  ValImgs_basedir = '../rawdata/HPA/ValImgs'
  if 'vit' in MODE.lower():
      Imgs_mean_std = '.../rawdata/TrainImgsHPA224_Mean_Std.mat'
  else:
      Imgs_mean_std = '../rawdata/TrainImgsHPA_Mean_Std.mat'
