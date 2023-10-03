from PIL import Image
import numpy as np
from torchvision import transforms as tfs
import matplotlib.pylab as plt
import cv2

augment_operat = tfs.Compose([
    tfs.RandomHorizontalFlip(0.5),
    tfs.RandomVerticalFlip(0.5),
    tfs.RandomRotation(180,expand=True),
    tfs.Resize([256,256]),
    tfs.RandomResizedCrop(256)]
)

def data_augment(data):
    a = Image.fromarray((data*255).astype(np.uint8)) ###
    data_tfs = augment_operat(a)
    return np.array(data_tfs).astype(np.float32)