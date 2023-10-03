import torch
from options import TrainOptions
from model_cl import DivCo_PLGAN
from saver import Saver,evaluate
import os
from data_reader import data_reader
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import time
from DataAugment import data_augment

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def main():
    # parse options
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.amp)
    print(torch.cuda.amp.autocast)

    parser = TrainOptions()
    opts = parser.parse()
    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    # model
    print('\n--- load model ---')
    model = DivCo_PLGAN(opts)

    if opts.checkpoint == -1:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        model_dir = os.path.join(opts.modeldir + '/' + opts.modelsname, '%05d.pth' % opts.checkpoint)
        print('Load Trained Model-%05d.pth' % opts.checkpoint)
        ep0, total_it = model.resume(model_dir)  ###
    ep0 = ep0+1
    model.setgpu()
    print('start the training at epoch %d'%(ep0))
    # saver for display and output
    transform = None #data_augment
    train_dataset = data_reader(is_Train=True,transform=transform)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=opts.batch_size,
        drop_last=True,
        num_workers=opts.workers,
        pin_memory=True,
    )

    valid_dataset = data_reader(is_Train=False)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=RandomSampler(valid_dataset),
        batch_size=opts.batch_size,
        drop_last=True,
        num_workers=opts.workers,
        pin_memory=True
    )
    saver = Saver(opts)
    evaluator = evaluate(opts)

    # train
    print('\n--- train ---')
    for ep in range(ep0, opts.max_epoch):
        np.random.seed(ep)
        torch.manual_seed(ep)
        torch.cuda.manual_seed_all(ep)
        end = time.time()

        iter, trainssim, trainmse, trainpsnr, trainlabelAcc = train_(train_loader, model, ep, evaluator)
        with torch.no_grad():
            validssim, validmse, validpsnr, validlabelAcc,test_inputs, test_labels = val_(valid_loader, model,ep,evaluator)

        print('\r', end='', flush=True)
        # save logs
        saver.write_img((ep + 1) * iter + 1, model,test_inputs, test_labels)
        saver.write_logs((ep + 1) * iter + 1, model,trainssim,trainpsnr)
        evaluator.write_test_logs((ep + 1) * iter + 1,test_inputs, test_labels,model,validssim,validpsnr)
        print('Train: SSIM %06f, MSE %06f, PSNR %06f, LabelAcc %06f' %(trainssim, trainmse, trainpsnr, trainlabelAcc))
        print('Test : SSIM %06f, MSE %06f, PSNR %06f, LabelAcc %06f'%(validssim, validmse, validpsnr, validlabelAcc))
        print('================================================================= %3.1f min \n'%((time.time() - end)/60))
        # Save network weights
        if ep > 18:
          saver.write_model(ep, (ep + 1) * iter + 1, model)
        else:
          if (ep%opts.save_step)==0:
            saver.write_model(ep, (ep + 1) * iter + 1, model)
    return

def train_(train_loader, model,epoch,evaluator):
    model.train()
    batch_time = AverageMeter()
    sum_iter = len(train_loader)
    end = time.time()
    data_time = AverageMeter()
    print_freq = 1
    Acc = AverageMeter()
    mse = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    for iter, iter_data in enumerate(train_loader, 0):
        data_time.update(time.time() - end)
        inputs, labels = iter_data
        # update model
        model.update(epoch, iter, sum_iter, inputs, labels)
        batch_time.update(time.time() - end)
        end = time.time()
        gloss = getattr(model, 'loss_G').cpu().detach().numpy()
        dloss = getattr(model, 'loss_D').cpu().detach().numpy()

        trainssim, trainmse, trainpsnr, trainlabelAcc = evaluator.train_evaluate(epoch, inputs, labels, model)
        Acc.update(trainlabelAcc, n=len(labels))
        psnr.update(trainpsnr, n=len(labels))
        mse.update(trainmse, n=len(labels))
        ssim.update(trainssim, n=len(labels))

        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == sum_iter:
            print('\r%5.1f   %5d    Glr %0.6f   Dlr %0.6f| Gloss %0.4f Dloss %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / sum_iter, iter + 1, model.gen_opt.param_groups[0]['lr']\
                       , model.dis_opt.param_groups[0]['lr'], gloss, dloss),end='', flush=True)
    return iter, ssim.avg, mse.avg, psnr.avg, Acc.avg

def val_(valid_loader, model,epoch,evaluator):
    Acc = AverageMeter()
    mse = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    model.eval()
    for iter, iter_data in enumerate(valid_loader, 0):
        inputs, labels = iter_data
        # update model
        validssim, validmse, validpsnr, validlabelAcc = evaluator.valid_evaluate(epoch, inputs, labels, model)
        Acc.update(validlabelAcc, n=len(labels))
        psnr.update(validpsnr, n=len(labels))
        mse.update(validmse, n=len(labels))
        ssim.update(validssim, n=len(labels))
        if iter == 0:
            inputs_, labels_ = inputs, labels
            break

    return ssim.avg, mse.avg, psnr.avg, Acc.avg,inputs_, labels_


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
    main()
