
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from config.config_ import *
import math
import torch
import torch.nn.functional as F
from scipy.io import savemat, loadmat



def multi_class_acc(preds, targs, th=10e-2):
    pred = preds.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    idx0 = np.where(sum_ == 0.0)[0]
    if len(idx0) == 0:
        pred = np.divide(pred, np.concatenate([sum_, sum_], axis=1))
    else:
        for i in range(len(idx0) + 1):
            if i == 0:
                div_ = sum_[0:idx0[i]]
                pred[0:idx0[i]] = np.divide(pred[0:idx0[i]], np.concatenate([div_, div_], axis=1))
            elif i < len(idx0):
                div_ = sum_[idx0[i - 1] + 1:idx0[i]]
                pred[idx0[i - 1] + 1:idx0[i]] = np.divide(pred[idx0[i - 1] + 1:idx0[i]],
                                                          np.concatenate([div_, div_], axis=1))
            else:
                div_ = sum_[idx0[i - 1] + 1:len(sum_)]
                pred[idx0[i - 1] + 1:len(sum_)] = np.divide(pred[idx0[i - 1] + 1:len(sum_)],
                                                            np.concatenate([div_, div_], axis=1))
    pred = pred.T
    targ = targs.cpu().detach().numpy().T
    diff = np.abs(pred-targ)
    diff_ = np.all(diff <= th,axis=0)
    return diff_.mean(),diff_


def normlize(preds, targs):
    pred = preds.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    # pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
    idx0 = np.where(sum_ == 0.0)[0]
    if len(idx0) == 0:
        pred = np.divide(pred, np.concatenate([sum_, sum_ ], axis=1))
    else:
        for i in range(len(idx0) + 1):
            if i == 0:
                div_ = sum_[0:idx0[i]]
                pred[0:idx0[i]] = np.divide(pred[0:idx0[i]], np.concatenate([div_, div_], axis=1))
            elif i < len(idx0):
                div_ = sum_[idx0[i - 1] + 1:idx0[i]]
                pred[idx0[i - 1] + 1:idx0[i]] = np.divide(pred[idx0[i - 1] + 1:idx0[i]],
                                                          np.concatenate([div_, div_], axis=1))
            else:
                div_ = sum_[idx0[i - 1] + 1:len(sum_)]
                pred[idx0[i - 1] + 1:len(sum_)] = np.divide(pred[idx0[i - 1] + 1:len(sum_)],
                                                            np.concatenate([div_, div_], axis=1))
    targ = targs.cpu().detach().numpy()
    return targ,preds


# def multi_class_acc(preds, targs, th=5e-2):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     diff = np.abs(pred[:,CATE]-targ[:,CATE])
#     diff_ = diff <= th
#     return diff_.mean(),diff_
#
# def person_corr(preds, targs):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     personCorr = []
#     for i in range(len(pred)):
#         personCorr_ = np.corrcoef(pred[i,CATE], targ[i,CATE])[0,1]
#         personCorr.append(personCorr_)
#     personCorr = np.array(personCorr)
#     return personCorr.mean(),personCorr
#
# def mse_(preds, targs):
#     pred = preds.cpu().detach().numpy()[:,CATE]
#     targ = targs.cpu().detach().numpy()[:,CATE]
#     diff_ = (pred-targ)**2
#     return diff_.mean(),diff_

# def sing_acc(preds, targs):

