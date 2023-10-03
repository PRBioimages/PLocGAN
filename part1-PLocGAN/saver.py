import os
import cv2
from imageio import imsave, imread
import numpy as np
from tensorboardX import SummaryWriter
import evaluate_model as evalmodel


def save_image(opts, imgs, inputs, epoch):
    height = opts.height
    batch_size = len(imgs)
    imgs_test_folder = opts.ImgsVerifydir
    if not os.path.exists(imgs_test_folder):
        os.makedirs(imgs_test_folder)

    temp_test_dir = os.path.join(imgs_test_folder, 'epoch_%d_#img.png' % (epoch))

    res = np.zeros((height * batch_size + 2 * (batch_size - 1), height * 4 + 6, 3))
    for k in range(batch_size):
        res[height * k + 2 * k:height * (k + 1) + 2 * k,0:height, 2] = inputs[k, :, :, 0] # input
        res[height * k + 2 * k:height * (k + 1) + 2 * k,0:height, 0] = inputs[k, :, :, 2] # input
        res[height * k + 2 * k:height * (k + 1) + 2 * k,height + 2:height * 2 + 2, 1] = inputs[k, :, :, 1]  # ground truth
        res[height * k + 2 * k:height * (k + 1) + 2 * k,height * 2 + 4:height * 3 + 4, 1] = imgs[k, :, :, 0]  # test1
        res[height * k + 2 * k:height * (k + 1) + 2 * k,height * 3 + 6:height * 4 + 6, 1] = cv2.absdiff(imgs[k, :, :, 0]\
                                                                                                                ,inputs[k, :, :, 1])
    imsave(temp_test_dir, (res * 255).astype(np.uint8))
    print("Evaluation images generated！==============================")

class Saver():
    def __init__(self, opts):
        self.opts = opts
        self.img_save_freq = 1
        self.model_save_freq = opts.save_step
        self.model_dir = os.path.join(opts.modeldir, opts.modelsname)
        self.logdir = os.path.join(opts.logdir, opts.logsname)


        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(opts.sampledir):
            os.makedirs(opts.sampledir)


        self.writer = SummaryWriter(log_dir=self.logdir)

    def write_logs(self, total_it, model,ssim,psnr):
        self.writer.add_scalar('/g_loss_content', getattr(model, 'loss_content'), total_it)
        self.writer.add_scalar('/g_loss_label', getattr(model, 'loss_G_label'), total_it)
        self.writer.add_scalar('/g_loss_adv', getattr(model, 'loss_G_GAN'), total_it)
        self.writer.add_scalar('/g_loss_contra', getattr(model, 'loss_contra'), total_it)
        self.writer.add_scalar('/g_loss', getattr(model, 'loss_G'), total_it)
        self.writer.add_scalar('/d_loss', getattr(model, 'loss_D'), total_it)
        self.writer.add_scalar('/d_loss_adv', getattr(model, 'loss_D_GAN'), total_it)
        self.writer.add_scalar('/d_loss_real_adv', getattr(model, 'loss_D_real_GAN'), total_it)
        self.writer.add_scalar('/d_loss_fake_adv', getattr(model, 'loss_D_fake_GAN'), total_it)
        self.writer.add_scalar('/d_loss_label', getattr(model, 'loss_D_label'), total_it)
        self.writer.add_scalar('/ssim', ssim, total_it)
        self.writer.add_scalar('/psnr', psnr, total_it)
        imgs,_,real_image,_ = model.assemble_outputs()
        for i in range(len(real_image)):
            self.writer.add_image('/gen_Image'+str(i), imgs[i,:,:,:], total_it)
            self.writer.add_image('/raw_Image'+str(i), real_image[i,:,:,:], total_it)

    def write_img(self, total_it, model, inputs, labels):
        imgs, _ = model.test_forward(inputs, labels)
        inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
        save_image(self.opts, imgs, inputs, total_it)

    # save model
    def write_model(self, ep, total_it, model):
        print('--- save the model @ ep %d ---' % (ep))
        model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)

class evaluate():
    def __init__(self, opts):
        self.opts = opts
        self.logdir = os.path.join(self.opts.logdir, self.opts.logs_val_name)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.writer = SummaryWriter(log_dir=self.logdir)

    def valid_evaluate(self, inputs, labels, model, thr=1e-3):
        imgs, fakeimgLabel = model.test_forward(inputs, labels)
        inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
        _, ssim = evalmodel.cal_ssim(inputs, imgs, len(labels), labels,singley=False)
        _, mse = evalmodel.cal_mse(inputs, imgs, len(labels), labels,singley=False)
        _, psnr = evalmodel.cal_psnr(inputs, imgs, len(labels), labels,singley=False)
        _, labelAcc = evalmodel.labelAcc(fakeimgLabel, labels, thr, len(labels))
        return ssim, mse, psnr, labelAcc

    def train_evaluate(self, inputs, labels, model, thr=1e-3):
        inputs = inputs.permute(0,2,3,1).cpu().detach().numpy()
        _, imgs, _, fakeimgLabel = model.assemble_outputs()
        _, ssim = evalmodel.cal_ssim(inputs, imgs, len(labels), labels, singley=False)
        _, mse = evalmodel.cal_mse(inputs, imgs, len(labels), labels, singley=False)
        _, psnr = evalmodel.cal_psnr(inputs, imgs, len(labels), labels, singley=False)
        _, labelAcc = evalmodel.labelAcc(fakeimgLabel, labels, thr, len(labels))
        return ssim, mse, psnr, labelAcc

    def write_test_logs(self,total_it, inputs, labels,model,ssim,psnr):
        model.test_forward(inputs, labels,enLoss=True)
        self.writer.add_scalar('/g_loss_content', getattr(model, 'loss_val_content'), total_it)
        self.writer.add_scalar('/g_loss_label', getattr(model, 'loss_val_G_label'), total_it)
        self.writer.add_scalar('/g_loss_adv', getattr(model, 'loss_val_G_GAN'), total_it)
        self.writer.add_scalar('/g_loss_contra', getattr(model, 'loss_val_contra'), total_it)
        self.writer.add_scalar('/g_loss', getattr(model, 'loss_val_G'), total_it)
        self.writer.add_scalar('/d_loss', getattr(model, 'loss_val_D'), total_it)
        self.writer.add_scalar('/d_loss_adv', getattr(model, 'loss_val_D_GAN'), total_it)
        self.writer.add_scalar('/d_loss_real_adv', getattr(model, 'loss_val_D_real_GAN'), total_it)
        self.writer.add_scalar('/d_loss_fake_adv', getattr(model, 'loss_val_D_fake_GAN'), total_it)
        self.writer.add_scalar('/d_loss_label', getattr(model, 'loss_val_D_label'), total_it)
        self.writer.add_scalar('/ssim', ssim, total_it)
        self.writer.add_scalar('/psnr', psnr, total_it)





