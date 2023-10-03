# coding: utf-8
import sys
sys.path.insert(0, '..')
import argparse
import pandas as pd
from scipy.io import loadmat
from scipy.io import savemat
# from scipy import stats
import  cv2
import matplotlib.pyplot as plt

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.autograd import Variable


from config.config_ import *
from utils.common_util import *
from networks.imageclsnet import init_network
from datasets.dataReader_test_ import data_reader
from utils.log_util import Logger
from utils.evalue_util import * # HPA dataset
from utils.evalue_util_real import * # real dataset
from utils.augment_util_ import *

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--out_dir', type=str,default='densenet_DA_p55_Bat32_',
                    help='destination where trained network should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for predicting (default: 0)')
parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--num_classes', default=5, type=int, help='number of classes (default: 28)')
parser.add_argument('--in_channels', default=3, type=int, help='in channels (default: 4)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--predict_epoch', default=None, type=int, help='number epoch to predict')

augment_list = ['default', 'flipud', 'fliplr','transpose', 'flipud_lr',
                'flipud_transpose', 'fliplr_transpose', 'flipud_lr_transpose']


def main(ReN):
    args = parser.parse_args()
    basedir = '../resultsLa/'
    log_out_dir = opj(basedir+'logs', 'Testlog_' + args.out_dir+str(ReN))
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.submit.txt'), mode='a')

    args.predict_epoch = 'final' if args.predict_epoch is None else '%03d' % args.predict_epoch
    network_path = opj(basedir+'models', 'model_' + args.out_dir+str(ReN), '%s.pth' % args.predict_epoch)
    submit_out_dir = opj(basedir+'submissions', 'submissions_' + args.out_dir+str(ReN),
                         'epoch_%s' % args.predict_epoch)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(submit_out_dir))
    if not ope(submit_out_dir):
        os.makedirs(submit_out_dir)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model = init_network(model_params)

    log.write(">> Loading network:\n>>>> '{}'\n".format(network_path))
    checkpoint = torch.load(network_path)
    model.load_state_dict(checkpoint['state_dict'])
    log.write(">>>> loaded network:\n>>>> epoch {}\n".format(checkpoint['epoch']))

    # moving network to gpu and eval mode
    model = DataParallel(model)
    model.cuda()
    model.eval()

    # Data loading code
    Images_Norm_params = {}
    if ope(Imgs_mean_std):
        matdata = loadmat(Imgs_mean_std)
        mean = matdata['Mean']
        std = matdata['Std']
    else:
        raise (FileNotFoundError("The file {} is not found!".format(Imgs_mean_std)))
    Images_Norm_params['in_channels'] = args.in_channels
    Images_Norm_params['mean'] = mean
    Images_Norm_params['std'] = std

    # Data loading code
    test_dataset = data_reader(MODE,Images_Norm_params)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    dataset = 'test'
    suffix = 'TestHpa_Comb'+str(COMB)+'Trial'+str(ReN)
    for augment in augment_list:
        test_loader.dataset.transform = eval('augment_%s' % augment)
        sub_submit_out_dir = opj(submit_out_dir,suffix, augment)
        results_dir = opj('../resultsLa/results',suffix, augment)
        if not ope(sub_submit_out_dir):
            os.makedirs(sub_submit_out_dir)
        if not ope(results_dir):
            os.makedirs(results_dir)
        with torch.no_grad():
            predict(test_loader, model, sub_submit_out_dir,results_dir, dataset)

def predict(test_loader, model, submit_out_dir,results_dir, dataset):
    accuracy = AverageMeter()
    personCorr = AverageMeter()
    MSE_ = AverageMeter()
    all_probs = []
    all_logits = []
    all_features = []
    all_labels = []
    accs = []
    corrs = []
    mses = []
    for it, iter_data in enumerate(test_loader, 0):
        images, labels = iter_data
        with torch.no_grad():
            # images = Variable(images.cuda(), volatile=True)
            images = Variable(images.cuda())
        outputs, features = model(images, mode='test')
        logits = outputs

        probs = torch.sigmoid(logits).data
        all_probs += probs.cpu().numpy().tolist()
        all_logits += logits.cpu().numpy().tolist()
        all_features += features.cpu().numpy().tolist()
        all_labels += labels.cpu().numpy().tolist()

        acc, accs_ = multi_class_acc(probs, labels)
        corr, corrs_ = person_corr(probs, labels)
        mse, mses_ = mse_(probs, labels)
        accs = np.append(accs, accs_, axis=0)
        corrs = np.append(corrs, corrs_, axis=0)
        mses = np.append(mses, mses_, axis=0)
        accuracy.update(acc.item(), n=len(accs_))
        personCorr.update(corr.item(), n=len(accs_))
        MSE_.update(mse.item(), n=len(accs_))
        print('\r%5d  |  %0.4f %0.4f %0.4f |  %0.4f %0.4f %0.4f | ... ' % (it + 1, accuracy.avg, personCorr.avg, MSE_.avg,\
                                                                           np.std(accs), np.std(corrs), np.std(mses)),end='', flush=True)

    all_probs = np.array(all_probs).reshape(len(mses), -1)
    all_logits = np.array(all_logits).reshape(len(mses), -1)
    all_features = np.array(all_features).reshape(len(mses), -1)

    np.save(opj(submit_out_dir, 'prob_%s.npy' % dataset), all_probs)
    np.save(opj(submit_out_dir, 'logit_%s.npy' % dataset), all_logits)
    np.save(opj(submit_out_dir, 'features_%s.npy' % dataset), all_features)

def prob_to_result(probs, img_ids, th=0.5):
    probs = probs.copy()
    probs[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1

    pred_list = []
    for line in probs:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)
    result_df = pd.DataFrame({'Id': img_ids, 'Predicted': pred_list})
    return result_df

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    for i in range(5):
        main(i)
    print('\nsuccess!')
