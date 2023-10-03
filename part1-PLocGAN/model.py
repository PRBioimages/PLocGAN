import networks as networks
import numpy as np
import torch
import torch.nn as nn

from config import IMGSIZE
# from ssim_torch import ssim
import torch.nn.functional as F



class DivCo_PLGAN(nn.Module):
    def __init__(self, opts):
        super(DivCo_PLGAN, self).__init__()
        # parameters
        lr = 1e-3
        self.eps = 1e-8
        self.nz = opts.hiddenz_size
        self.opt = opts
        self.class_num = opts.n_class
        self.E = networks.Encoder(opts).cuda() ###
        self.G = networks.generator(opts).cuda()###
        self.D = networks.discriminator().cuda()

        self.enc_opt = torch.optim.Adam(self.E.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        self.dis_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        self.enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.enc_opt, \
                                                                        T_max=self.opt.lr_decay_iter, eta_min=self.opt.learning_rate)
        self.gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.gen_opt,
                                                                        T_max=self.opt.lr_decay_iter, eta_min=self.opt.learning_rate)
        self.dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.dis_opt, \
                                                                        T_max=self.opt.d_lr_decay_iter, eta_min=self.opt.d_learning_rate)

        self.BCE_loss = torch.nn.BCELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()


    def initialize(self):
        self.E.weight_init()
        self.G.weight_init()
        self.D.weight_init()

    def setgpu(self):
        self.E.cuda()
        self.D.cuda()
        self.G.cuda()
        # self.E = torch.nn.DataParallel(self.E,device_ids=[0,1]).cuda()
        # self.G = torch.nn.DataParallel(self.G,device_ids=[0,1]).cuda()
        # self.D = torch.nn.DataParallel(self.D,device_ids=[0,1]).cuda()

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_z_random(self, batchSize, nz):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z


    def forward(self):
        self.label_one_hot = self.label
        self.z_random = self.get_z_random(self.real_image.size(0), self.nz) #[32,100]
        self.z_r1, self.down_outputs1 = self.E.forward(self.ref_image)
        self.fake_image1 = self.G.forward(self.down_outputs1, self.z_random, self.z_r1, self.label_one_hot)

    def update_D(self):
        self.set_requires_grad(self.D, True)
        # update discriminator
        self.dis_opt.zero_grad()
        self.fake_image1_rs = torch.cat((self.ref_image,self.fake_image1),dim=1)
        real_image_rs = torch.cat((self.ref_image,self.real_image.view(-1, 1, IMGSIZE, IMGSIZE)), dim=1)

        # adv loss
        self.loss_D_GAN, self.loss_D_real_GAN, self.loss_D_fake_GAN =\
            self.backward_D(self.D, real_image_rs, self.fake_image1_rs, self.label_one_hot)

        # label loss
        all0 = torch.zeros_like(self.label_one_hot)
        all0 = all0.cuda()
        self.loss_D_label = self.compute_mse_loss(self.fake_label1, all0) + \
                                      self.compute_mse_loss(self.real_label1,self.label_one_hot)

        self.loss_D = self.loss_D_label + self.loss_D_GAN
        self.loss_D.backward(retain_graph=True)
        self.dis_opt.step()
        if self.iter <= self.opt.d_lr_decay_iter:
            self.dis_scheduler.step()


    def update_G(self):
        self.set_requires_grad(self.D, False)
        # update generator
        self.gen_opt.zero_grad()
        self.enc_opt.zero_grad()

        advloss1, labelloss1 = self.backward_G(self.D, self.fake_image1_rs, self.label_one_hot)

        # adv loss
        self.loss_G_GAN = advloss1
        # content loss
        mse = self.compute_mse_loss(self.fake_image1.view(-1,IMGSIZE,IMGSIZE),self.real_image)
        # ssim_ = (1-ssim(self.fake_image1,self.real_image.view(-1,1, IMGSIZE, IMGSIZE)))/2
        self.loss_content = mse# ssim_ #mmse
        # label loss
        self.loss_G_label = labelloss1

        self.loss_G = self.loss_G_GAN*self.opt.gamma_genL1 + self.loss_G_label*self.opt.gamma_genLabel+\
                      self.loss_content*self.opt.gamma_genMSE

        self.loss_G.backward()
        self.enc_opt.step()
        self.gen_opt.step()
        if self.iter <= self.opt.lr_decay_iter:
            self.enc_scheduler.step()
            self.gen_scheduler.step()

    def compute_contrastive_loss(self, feat_q, feat_k):
        out = torch.mm(feat_q, feat_k.transpose(1,0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))
        return loss

    def compute_mse_loss(self,fake,real):
        return torch.nn.functional.mse_loss(input=fake,target=real)

    def update(self, ep, it,sum_iter, image, label,cate_weight=None):
        self.cate_weight = cate_weight
        self.iter = ep * sum_iter + it + 1
        self.real_image = image[:, 1, :, :].cuda()
        self.ref_image = image[:, 0, :, :].view(-1, 1,IMGSIZE,IMGSIZE).cuda()
        self.label = label.cuda()
        self.forward()
        self.update_D()
        self.update_G()

    def backward_D(self, netD, real, fake, label):
        pred_fake, self.fake_label1 = netD.forward(fake)
        pred_real, self.real_label1 = netD.forward(real)
        all0 = torch.zeros_like(pred_fake).cuda()
        all1 = torch.ones_like(pred_real).cuda()
        ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all0)
        ad_true_loss = nn.functional.binary_cross_entropy_with_logits(pred_real, all1)
        loss_D = ad_true_loss + ad_fake_loss
        return loss_D,ad_true_loss,ad_fake_loss

    def backward_G(self, netD, fake, label):
        pred_fake,fake_label = netD.forward(fake)
        all_ones = torch.ones_like(pred_fake).cuda()
        loss_G = nn.functional.binary_cross_entropy_with_logits(pred_fake, all_ones)
        loss_label_G = self.compute_mse_loss(fake_label,self.label_one_hot)
        return loss_G,loss_label_G

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        self.D.load_state_dict(checkpoint['dis'])
        self.E.load_state_dict(checkpoint['enc'])
        self.G.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            self.enc_opt.load_state_dict(checkpoint['enc_opt'])
            self.dis_scheduler.load_state_dict(checkpoint['dis_sch'])
            self.gen_scheduler.load_state_dict(checkpoint['gen_sch'])
            self.enc_scheduler.load_state_dict(checkpoint['enc_sch'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
                'dis': self.D.state_dict(),
                'gen': self.G.state_dict(),
                'enc':self.E.state_dict(),
                'dis_opt': self.dis_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'enc_opt':self.enc_opt.state_dict(),
                'dis_sch':self.dis_scheduler.state_dict(),
                'gen_sch': self.gen_scheduler.state_dict(),
                'enc_sch': self.enc_scheduler.state_dict(),
                'ep': ep,
                'total_it': total_it
                    }
        torch.save(state, filename)
        return

    # for training set
    def assemble_outputs(self):
        image_fake1 = self.fake_image1.permute(0,2,3,1)
        real_image = self.real_image.view(-1,1,IMGSIZE,IMGSIZE)
        fakelabel1 = self.fake_label1.cpu().detach().numpy()
        return self.fake_image1,image_fake1.cpu().detach().numpy(),real_image,fakelabel1

    # for validation
    def test_forward(self, image, label, sample_z=0, inputz=False, enLoss=False):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_image = (image[:,0, :, :]).view(-1, 1,IMGSIZE,IMGSIZE).cuda()
        real_image = (image[:, 1, :, :]).cuda()
        label = label.cuda()
        if inputz:
            z_random = sample_z.cuda()
        else:
            z_random = self.get_z_random(real_image.size(0), self.nz).cuda()
        z_r, down_outputs = self.E.forward(ref_image)
        outputs = self.G.forward(down_outputs, z_random, z_r, label)
        fake_image_rs = torch.cat((ref_image,outputs),dim=1)
        pred_fake, fake_label = self.D.forward(fake_image_rs)
        if enLoss:
            real_image_rs = torch.cat((ref_image,real_image.view(-1, 1, IMGSIZE, IMGSIZE)),dim=1)

            pred_real, real_label = self.D.forward(real_image_rs)
            ## D_valid_loss
            all0 = torch.zeros_like(pred_fake).cuda()
            all1 = torch.ones_like(pred_real).cuda()
            self.loss_val_D_fake_GAN = nn.functional.binary_cross_entropy_with_logits(pred_fake, all0)
            self.loss_val_D_real_GAN = nn.functional.binary_cross_entropy_with_logits(pred_real, all1)
            self.loss_val_D_GAN = self.loss_val_D_real_GAN + self.loss_val_D_fake_GAN
            tarlabel = torch.zeros_like(label)
            tarlabel = tarlabel.cuda()
            self.loss_val_D_label = self.compute_mse_loss(fake_label, tarlabel) + \
                                    self.compute_mse_loss(real_label, label)
            self.loss_val_D = self.loss_val_D_label + self.loss_val_D_GAN

            ## G_valid_loss
            all_ones = torch.ones_like(pred_fake).cuda()
            advloss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all_ones)
            labelloss = self.compute_mse_loss(fake_label, label)
            self.loss_val_G_GAN = advloss
            # content loss
            mse = self.compute_mse_loss(outputs.view(-1, IMGSIZE, IMGSIZE),real_image)
            # ssim_ = (1 - ssim(outputs,real_image.view(-1,1,IMGSIZE, IMGSIZE)))
            self.loss_val_content = mse#ssim_# mse
            # label loss
            self.loss_val_G_label = labelloss
            self.loss_val_G = self.loss_val_G_GAN * self.opt.gamma_genL1 + self.loss_val_G_label * self.opt.gamma_genLabel + \
                              self.loss_val_content * self.opt.gamma_genMSE

        return outputs.permute(0, 2, 3, 1).cpu().detach().numpy(), fake_label.cpu().detach().numpy()

    # for test set
    def test_forward_(self, image, label,sample_z=0,inputz=False):
        ref_image = (image[:,0, :, :]).view(-1, 1,IMGSIZE,IMGSIZE).cuda()
        label = label.cuda()
        real_image = (image[:, 1, :, :]).cuda()
        # rand z
        if inputz:
            z_random = sample_z.cuda()
        else:
            z_random = self.get_z_random(real_image.size(0), self.nz)
        z_r, down_outputs = self.E.forward(ref_image)
        fake_image = self.G.forward(down_outputs, z_random, z_r, label)
        outputs = fake_image[:real_image.size(0)]
        fake_image_rs = torch.cat((ref_image,outputs), dim=1)
        pred_fake, fake_label = self.D.forward(fake_image_rs)
        return outputs.permute(0, 2, 3, 1).cpu().detach().numpy(), fake_label.cpu().detach().numpy()




