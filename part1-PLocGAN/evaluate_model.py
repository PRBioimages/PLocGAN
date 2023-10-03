import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from PIL import Image
import lpips
from fid_score.fid_score import FidScore
from prdc import compute_prdc
from torchvision import models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
import math
import os
import cv2

img_to_tensor = transforms.ToTensor()
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def create_csv(path, csv_head):
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)


def write_csv(data_row, path):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
        # print("write over")

def labelAcc(fakeimgLabel, labels, thr,batch_size):
    TP = 0
    for i in range(batch_size):
        tmplabel = labels[i]
        tmpfakelabel = fakeimgLabel[i]
        dis = ((tmplabel[0] - tmpfakelabel[0]) ** 2 + (tmplabel[1] - tmpfakelabel[1]) ** 2) / 2
        if dis <= thr:
            TP = TP + 1
    acc = TP / batch_size
    return TP, acc


def cal_ssim(im1, im2):
    value = []
    for i in range(len(im1)):
        imgreal = im1[i, :, :, 1]
        imgfake = im2[i, :, :, 0]
        value.append(ssim(imgreal, imgfake,data_range=1))
    return value, np.mean(value)

def evaluate_lpips(img1,img2):

    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()

    img1_tens = TF.to_tensor(img1).unsqueeze(0) * 2 - 1
    img2_tens = TF.to_tensor(img2).unsqueeze(0) * 2 - 1

    img1_tens = img1_tens.to('cuda')
    img2_tens = img2_tens.to('cuda')

    lpipsmetric = loss_fn_vgg.forward(img1_tens, img2_tens) ## data range [-1,1]
    print(np.squeeze(lpipsmetric.cpu().detach().numpy()))
    return np.squeeze(lpipsmetric.cpu().detach().numpy())

def evaluate_fid(paths,batch_size):
    device = torch.device('cuda:0')
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()
    return score

def psnr(tf_img1, tf_img2, max_val):
    MSEimg = mse(tf_img1, tf_img2)
    psnr = 20 * math.log10(max_val / math.sqrt(MSEimg))
    return psnr

def cal_psnr(im1, im2):
    value = []
    for i in range(len(im1)):
        imgreal = im1[i, :, :, 1]
        imgfake = im2[i, :, :, 0]
        value.append(psnr(imgreal, imgfake,max_val=1))
    return value, np.mean(value)

def extract_features(img):
    net = models.vgg16(pretrained=True).to('cuda')
    net.eval()
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img3d = img.reshape(1, 224, 224).repeat(3, axis=0)
    input = img_to_tensor(img3d)  # image to tensor
    input = input.reshape(1, 3, 224, 224).float().cuda()
    ifeatures = net.features(input)
    ifeatures = net.avgpool(ifeatures)
    ifeatures = torch.flatten(ifeatures, 1)
    features = net.classifier[:4](ifeatures).cpu().detach().numpy()
    return features

def cal_coverage_density(real_features,fake_features,nearest_k):
    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=nearest_k)
    coverage = metrics['coverage']
    density = metrics['density']
    return coverage,density

def calc_lpips(opts,data,model):
    samples = []
    labels = []
    lpipssum = []
    iter = 0
    for k, iter_data in enumerate(data, 0):
        if k==80:
            break
        # torch.cuda.empty_cache()
        print('epoch:%d'%k)
        x, y = iter_data

        for i in range(10):
            output_test,_ = model.test_forward_(x, y)
            if i == 0:
                samples = output_test
            else:
                samples = np.append(samples, output_test, axis=0)
            labels.extend(y)
        for imgnum in range(opts.batch_size):
            calcdata = samples[np.array(range(10))*opts.batch_size+imgnum,:,:]
            ilpips = 0
            for n in range(10):
                for m in range(n+1,10):
                    tmplpips = evaluate_lpips(calcdata[n,:,:],calcdata[m,:,:])
                    if iter == 0:
                        create_csv(opts.test_Metric,csv_head=["Iter","Epoch","Idx", "LPIPS"])
                    data_row = [str(iter),str(k),str(ilpips), str(tmplpips)]
                    write_csv(data_row, opts.test_Metric)
                    iter = iter + 1
                    ilpips = ilpips + 1
                    lpipssum.append(tmplpips)
                    img1 = np.zeros((opts.height, opts.height, 3))
                    img1[:, :, 1] = calcdata[n, :, :,0]
                    img2 = np.zeros((opts.height, opts.height, 3))
                    img2[:, :, 1] = calcdata[m, :, :,0]
        print('=========================================LPIPS %04f'%np.mean(lpipssum))
    return np.mean(lpipssum)


def diversity_genimg(opts,data,suffix=None):
    sum_store_path = os.path.join('../Fidrealimg/'+opts.comb+'_'+suffix,'Sum')
    if not os.path.exists(sum_store_path):
        os.makedirs(sum_store_path)
    i = 0
    for iter, iter_data in enumerate(data, 0):
        inputs, labels = iter_data
        inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
        for n in range(len(labels)):
            res2 = np.zeros((opts.height, opts.height,3))
            res2[:, :, 1] = inputs[n, :, :,1]
            cv2.imwrite(os.path.join(sum_store_path, 'Fidreal_epoch%d_batch%d_img.png') % (iter, n),
                        (res2 * 255).astype(np.uint8))
            i = i + 1

def calc_fid(opts,result_dir,suffix=None):
    gen_batch = 2 * opts.batch_size
    ireal = os.path.join('../Fidrealimg/'+opts.comb+'_'+suffix,'Sum')
    ifake = os.path.join(result_dir, 'Sum_'+opts.suffix)
    path = [ifake, ireal]
    fid = evaluate_fid(path, gen_batch)
    return fid
