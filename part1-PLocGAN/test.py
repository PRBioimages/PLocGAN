import torch
from model_cl import DivCo_PLGAN
from saver import *
import os
import cv2
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler,RandomSampler
from data_reader_test import data_reader
import evaluate_model
from scipy.io import savemat,loadmat
# from gradcam import *
from PIL import Image
import argparse
import pandas as pd
from tool import *

workdirec = './'
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.namestr = '_cl_CYNP'
        self.parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 3)')
        self.parser.add_argument('--modeldir', type=str, default=workdirec+'modeldir' + self.namestr, help='The stored model dir ')
        self.parser.add_argument('--sampledir', type=str, default=workdirec+'sampledir' + self.namestr, help='Many others stored dir ')
        self.parser.add_argument('--merge_dir', type=str, default=workdirec + '/mergedir' + self.namestr,
                                 help='Many others stored dir ')
        self.parser.add_argument('--modelsname', type=str, default='model', help='folder name to save outputs')
        self.parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--d_learning_rate', type=float, default=4e-4, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr_decay_iter', type=int, default=150000,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--d_lr_decay_iter', type=int, default=100000,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--lr_min_iter', type=int, default=500000,
                                 help='the iter set for min learning rate(1e-4) ')

        self.parser.add_argument('--checkpoint', type=int, default=57, help='number of epochs to be reloaded') ## 1519
        self.parser.add_argument('--n_class', type=int, default=2, help='number of classes')
        self.parser.add_argument('--height', type=int, default=256, help='height of image')
        self. parser.add_argument('--width', type=int, default=256, help='width of image')
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size') # 10raw; 5; 2ref
        self.parser.add_argument('--hiddenz_size', type=int, default=16, help='size of the hidden z')
        self.parser.add_argument('--hiddenr_size', type=int, default=8, help='size of the hidden r')

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--comb', type=str, default='LyMo', help='phase for dataloading')
        self.parser.add_argument('--suffix', type=str, default='U2mlLa_', help='phase for dataloading')
        self.parser.add_argument('--num', type=int, default=6, help='number of outputs per image')
        self.parser.add_argument('--test_MetricSimp', type=str,
                                 default=os.path.join(workdirec + 'sampledir' + self.namestr, \
                                                      'test_metricSimp' + self.namestr + '.csv'), \
                                 help='Stored evaluated txt for Testing')
        self.parser.add_argument('--test_Metric', type=str,
                                 default=os.path.join(workdirec + 'sampledir' + self.namestr, \
                                                      'test_metric' + self.namestr + '.csv'), \
                                 help='Stored evaluated txt for Testing')
        self.parser.add_argument('--Mul_Label_GenerateImagedir', type=str, default=workdirec+'save_image_MulLabel'\
                                                                                   + self.namestr, help='saves results here.')
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt


def save_image(opts,num,imgs,output_test,idx2,label):

    if not os.path.exists(opts.merge_dir):
        os.makedirs(opts.merge_dir)

    realdir = opts.merge_dir+'/srrealdir'
    GIdir = opts.merge_dir + '/srGIRdir'
    fakedir = opts.merge_dir + '/srRefdir'
    if not os.path.exists(realdir):
        os.makedirs(realdir)
    if not os.path.exists(GIdir):
        os.makedirs(GIdir)
    if not os.path.exists(fakedir):
        os.makedirs(fakedir)

    for i in range(len(label)):
        num = num +1
        ilabel = label[i]

        # if idx2==0:
        #     temp_real_dir = os.path.join(realdir, 'epoch_%d_%dreal_z%d.jpg' % (ep,i,idx2))
        #     res = np.zeros((opts.height, opts.height, 3))
        #     res[:, :, 0] = imgs[i, :, :, 0]
        #     YImg = np.zeros([opts.height, opts.height, 3])
        #     YImg[:, :, 1] = imgs[i, :, :, 2]
        #     YImg[:, :, 2] = imgs[i, :, :, 2]
        #     res_ = Image.fromarray(np.uint8((res+YImg) * 255))
        #     res_.save(temp_real_dir, dpi=(300, 300))
        # res = ((res+YImg) * 255).astype(np.uint8)
        # cv2.imwrite(temp_real_dir, res)

        temp_GI_dir = os.path.join(GIdir, 'epoch_%d_%dGI_z%d.jpg' % (ep, i, idx2))
        res = np.zeros([opts.height, opts.height, 3])
        res[:, :, 1] = output_test[i, :, :, 0]
        res_ = Image.fromarray(np.uint8(res * 255))
        res_.save(temp_GI_dir, dpi=(300, 300))
        # imsave(temp_GI_dir, (res * 255).astype(np.uint8))
        # if idx2 == 0:
        #     temp_fake_dir = os.path.join(fakedir, 'epoch_%d_%dfake_z%d.jpg' % (ep, i, idx2))
        #     res = np.zeros((opts.height, opts.height, 3))
        #     res[:,:,0] = imgs[i, :, :, 0]
        #     res[:, :, 1] = imgs[i, :, :, 1]
        #     YImg = np.zeros([opts.height, opts.height, 3])
        #     YImg[:, :, 1] = imgs[i, :, :, 2]
        #     YImg[:, :, 2] = imgs[i, :, :, 2]
        #     res_ = Image.fromarray(np.uint8((res + YImg) * 255))
        #     res_.save(temp_fake_dir, dpi=(300, 300))
        # cv2.imwrite(temp_fake_dir, ((res+YImg) * 255).astype(np.uint8))
    return num

def sample_z(opts,batch,idx2):
    latent = float(idx2) / (opts.num - 1) * 2 - 1
    z_random = torch.ones(batch, opts.hiddenz_size) * latent
    return z_random


def evaluate_saveimage(opts,data,model):
    for idx2 in range(6):
        num = 0
        for iter, iter_data in enumerate(data, 0):
            print("===============", iter)
            imgs, label = iter_data
            sampled_zs = sample_z(opts, len(label), idx2)
            output_test, _ = model.test_forward_(imgs, label, sample_z=sampled_zs, inputz=True)
            num = save_image(opts, num, imgs.permute(0, 2, 3, 1).cpu().detach().numpy(), output_test, idx2,
                                 label.cpu().detach().numpy())

def evaluate_real(opts,data,model,result_dir):
    psnr = []
    ssim = []

    fake_features = np.zeros([len(data.dataset.labels), 4096])
    real_features = np.zeros([len(data.dataset.labels), 4096])
    fakeFeasdir = os.path.join(result_dir, 'Testfake_' + opts.comb + '_' + opts.suffix + '.mat')
    realFeasdir = os.path.join('../Fidrealimg/' + opts.comb + '_' + opts.suffix,
                                   'Test_' + opts.comb + '_' + opts.suffix + '.mat')
    num_iter = 0
    num_iter_ = 0
    for i, iter_data in enumerate(data, 0):
        print("===============", i)
        images, label = iter_data
        output_test,preds = model.test_forward_(images, label)
        imgs = images.permute(0, 2, 3, 1).cpu().detach().numpy()
        tmpssim, _ = evaluate_model.cal_ssim(imgs, output_test)
        tmppsnr, mean_psnr = evaluate_model.cal_psnr(imgs, output_test)
        if not os.path.exists(fakeFeasdir):
            for ibatch in range(len(label)):
                fakeimg = output_test[ibatch, :, :, 0]
                ifakefeatures = evaluate_model.extract_features(fakeimg)
                fake_features[num_iter, :] = ifakefeatures
                num_iter = num_iter + 1


        if not os.path.exists(realFeasdir):
            for ibatch in range(len(label)):
                realimg = imgs[ibatch, :, :, 1]
                irealfeatures = evaluate_model.extract_features(realimg)
                real_features[num_iter_, :] = irealfeatures
                num_iter_ = num_iter_ + 1
        ssim+= tmpssim.tolist()
        psnr += tmppsnr.tolist()

    if not os.path.exists(realFeasdir):
        savemat(realFeasdir, mdict={'real_features': real_features})
    else:
        matdata = loadmat(realFeasdir)
        real_feas = matdata['real_features']
        len_feas = len(fake_features)
        real_features = real_feas[:len_feas]

    if not os.path.exists(fakeFeasdir):
        savemat(fakeFeasdir, mdict={'fake_features': fake_features})
    else:
        matdata = loadmat(fakeFeasdir)
        fake_feas = matdata['fake_features']
        len_feas = len(fake_feas)
        fake_features = fake_feas[:len_feas]

    coverage, density = evaluate_model.cal_coverage_density(real_features, fake_features, nearest_k=10)

    savemat(os.path.join(result_dir,'test_results.mat'), mdict={'SSIM':ssim,'PSNR':psnr,'Coverage':coverage,'density':density})
    ssim = np.array(ssim)
    psnr = np.array(psnr)
    print("SSIM:(%.4f,%.4f)" % (np.mean(ssim), np.std(ssim)),"PSNR:(%.4f,%.4f)" % (np.mean(psnr), np.std(psnr)))
    print("Density:%.4f" % density, "Coverage:%.4f" % coverage)

def evaluate_fid(opts,data,model,result_dir):
    # calculate FID
    generate_img_z(opts, data, model, result_dir)
    fid = evaluate_model.calc_fid(opts, result_dir, suffix=opts.suffix)
    print('FID:%.4f'%fid)

def evaluate_diversity(opts,data,model):
    # # calculate LPIPS
    # # Ref: https://github.com/richzhang/PerceptualSimilarity
    genImg_lpips = evaluate_model.calc_lpips(opts,data,model)
    print('LPIPS:%.4f' % genImg_lpips)

def generate_img_z(opts,data,model,path):
    sum_store_path = os.path.join(path, 'Sum_'+opts.suffix)
    if not os.path.exists(sum_store_path):
        os.makedirs(sum_store_path)
    i = 0
    for iter, iter_data in enumerate(data, 0):
        inputs, labels = iter_data
        inputz = False
        for k in range(10):
            output_test,_ = model.test_forward_(inputs, labels, inputz)
            for n in range(len(labels)):
                res = np.zeros((256, 256, 3))
                res[:, :, 1] = output_test[n, :, :,0]
                cv2.imwrite(os.path.join(sum_store_path, 'epoch_%d_batch_%d_#img_%d.png') % (iter, n, k),
                            (res * 255).astype(np.uint8))
            i = i+1

def main(i,opts,dataloader):
    print('\n--- load model ---')
    model = DivCo_PLGAN(opts)
    model.eval()
    model.setgpu()
    model_dir = os.path.join(opts.modeldir + '/' + opts.modelsname, '%05d.pth' % i)
    print('Load Trained Model-%05d.pth' % i)
    model.resume(model_dir, train=False)

    # directory
    result_dir = opts.Mul_Label_GenerateImagedir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # test
    fid = 0
    print('\n--- testing ---')
    evaluate_model.diversity_genimg(opts, dataloader, opts.suffix)
    # evaluate_real(opts,dataloader,model,result_dir)
    evaluate_fid(opts,dataloader,model,result_dir)
    # evaluate_diversity(opts,dataloader,model)
    return fid

if __name__ == '__main__':
    parser = TestOptions()
    opts = parser.parse()
    test_dataset = data_reader()
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=opts.batch_size,
        drop_last=False,
        num_workers=opts.workers,
        pin_memory=True,
    )
    fid = main(opts.checkpoint,opts,test_loader)