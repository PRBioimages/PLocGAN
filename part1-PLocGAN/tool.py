import os
import math
import numpy as np
import cv2
import torch

def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    if len(image.shape) == 3 :
        image = np.transpose(image, (1, 2, 0))
    image = image*std + mean
    image = image.astype(dtype=np.uint8)
    return image

def tensor_to_label(tensor):
    label = tensor.numpy()
    label = label.astype(dtype=np.uint8)
    return label

def image_to_tensor(image):
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor

def label_to_tensor(label):
    tensor = torch.from_numpy(label).type(torch.FloatTensor)
    return tensor


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


