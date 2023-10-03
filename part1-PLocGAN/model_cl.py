import os
import networks_cl as networks
import numpy as np
import torch
import torch.nn as nn



class DivCo_PLGAN(nn.Module):
    def __init__(self, opts):
        super(DivCo_PLGAN, self).__init__()
        # parameters
        lr = 0.001
        self.nz = opts.hiddenz_size
        self.opt = opts
        self.class_num = opts.n_class
        self.E = networks.Encoder(opts)
        self.G = networks.generator(opts)
        self.D = networks.discriminator()

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
        self.E = self.E.cuda()
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        # self.E = torch.nn.DataParallel(self.E).cuda()
        # self.G = torch.nn.DataParallel(self.G).cuda()
        # self.D = torch.nn.DataParallel(self.D).cuda()


    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z

    def latent_augmented_sampling(self,query):
        pos = torch.cuda.FloatTensor(query.shape).uniform_(-self.opt.radius, self.opt.radius).add_(query)
        negs = []
        for k in range(self.opt.num_negative):
            neg = self.get_z_random(self.real_image.size(0), self.nz, 'gauss')
            while (neg-query).abs().min() < self.opt.radius:
                neg = self.get_z_random(self.real_image.size(0), self.nz, 'gauss')
            negs.append(neg)
        return query, pos, negs

    def forward(self):
        self.label_one_hot = self.label
        query = self.get_z_random(self.real_image.size(0), self.nz, 'gauss')
        _, pos, negs = self.latent_augmented_sampling(query)
        self.z_random = [query, pos] + negs
        z_conc = torch.cat(self.z_random,dim=0)

        self.label_conc = torch.cat([self.label_one_hot] * (self.opt.num_negative+2),0)
        self.z_r,self.down_outputs = self.E.forward(self.ref_image)
        z_r_conc = torch.cat([self.z_r]* (self.opt.num_negative+2),0)
        down_outputs_conc = [x.repeat(self.opt.num_negative + 2, 1, 1, 1) for x in self.down_outputs]
        self.fake_image = self.G.forward(down_outputs_conc,z_conc,z_r_conc,self.label_conc)

        self.fake_image1 = self.fake_image[:self.real_image.size(0)]

    def update_D(self):
        self.set_requires_grad(self.D, True)
        # update discriminator
        self.dis_opt.zero_grad()
        self.fake_image_rs = torch.cat((self.ref_image[:,0,:,:].view(self.opt.batch_size, 1,256,256),\
                                                 self.fake_image1,self.ref_image[:,1,:,:].view(self.opt.batch_size, 1,256,256)),dim=1)
        real_image_rs = torch.cat((self.ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256), \
                                                     self.real_image.view(self.opt.batch_size, 1, 256, 256)\
                                                   , self.ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)), dim=1)

        loss_D_GAN, loss_D_real_GAN, loss_D_fake_GAN = self.backward_D(self.D, real_image_rs, self.fake_image_rs, self.label_one_hot)
        # adv loss
        self.loss_D_GAN = loss_D_GAN
        self.loss_D_real_GAN = loss_D_real_GAN
        self.loss_D_fake_GAN = loss_D_fake_GAN

        # label loss
        all0 = torch.zeros_like(self.label_one_hot)
        all0 = all0.cuda()
        self.loss_D_label = self.compute_mse_loss(self.fake_label, all0) + \
                            self.compute_mse_loss(self.real_label, self.label_one_hot)

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
        self.contrastive_feats_D(self.fake_image, self.label_conc)

        advloss, labelloss = self.backward_G(self.D, self.fake_image_rs, self.label_one_hot)
        # adv loss
        self.loss_G_GAN = advloss
        # content loss
        self.loss_content = self.compute_mse_loss(self.fake_image1.view(self.opt.batch_size,256,256),self.real_image)
        # label loss
        self.loss_G_label = labelloss
        # contrastive loss
        self.loss_contra = 0.0
        for i in range(self.real_image.size(0)):
                logits_real = self.feats_real[i:self.feats_real.shape[0]:self.real_image.size(0)].view(self.opt.num_negative + 2,-1)
                if self.opt.featnorm:
                    logits_real = logits_real / torch.norm(logits_real, p=2, dim=1, keepdim=True)
                contra_real = self.compute_contrastive_loss(logits_real[0:1], logits_real[1:])
                contra = contra_real
                self.loss_contra += contra

        self.loss_G = self.loss_G_GAN*self.opt.gamma_genL1 + self.loss_contra * self.opt.gamma_genContra + \
                                  self.loss_G_label*self.opt.gamma_genLabel+self.loss_content*self.opt.gamma_genMSE
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

    def update(self, ep, it, sum_iter, image, label):
        self.iter = ep * sum_iter + it + 1
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_image = torch.from_numpy(image[:, :, :, 1]).cuda()
        self.ref_image = torch.from_numpy(image[:, :, :, (0, 2)]).permute(0, 3, 1, 2).cuda()
        self.label = torch.Tensor(label).cuda()
        self.forward()
        self.update_D()
        self.update_G()

    def backward_D(self, netD, real, fake, label):
        pred_fake, self.fake_label = netD.forward(fake)
        pred_real, self.real_label = netD.forward(real)
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
     # use features from discriminator to calculate cl loss
    def contrastive_feats_D(self, fake, label):
        ref1 = self.ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256)
        ref1s = ref1.repeat(self.opt.num_negative + 2, 1, 1, 1)
        ref2 = self.ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)
        ref2s = ref2.repeat(self.opt.num_negative + 2, 1, 1, 1)
        fake = torch.cat((ref1s, fake, ref2s), dim=1)
        _, _,self.feats_real = self.D.forward(fake, enc_feat=True)

    # use features from generator to calculate cl loss
    # def backward_E(self,fake):
    #     fake = torch.cat([fake,torch.zeros_like(fake).cuda()],dim=1)
    #     _, _, self.feats = self.E.forward(fake,enc_feat=True)

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir) ###
        self.D.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['dis'].items()})
        self.E.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['enc'].items()})
        self.G.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['gen'].items()})
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            for state in self.dis_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            for state in self.gen_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.enc_opt.load_state_dict(checkpoint['enc_opt'])
            for state in self.enc_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
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
                'dis_sch': self.dis_scheduler.state_dict(),
                'gen_sch': self.gen_scheduler.state_dict(),
                'enc_sch': self.enc_scheduler.state_dict(),
                'ep': ep,
                'total_it': total_it
                    }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        image_fake = self.fake_image.permute(0, 2, 3, 1)
        real_image = self.real_image.view(self.opt.batch_size, 1, 256, 256)
        fakelabel = self.fake_label.cpu().detach().numpy()
        return self.fake_image, image_fake.cpu().detach().numpy(), real_image, fakelabel

    def test_forward(self, image, label,sample_z=0,inputz=False,enLoss=False):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_image = torch.from_numpy(image[:, :, :, (0, 2)]).permute(0, 3, 1, 2).cuda()
        label = torch.Tensor(label).cuda()
        real_image = torch.from_numpy(image[:,:,:,1]).cuda()
        # rand z
        if inputz:
            z_random = sample_z
        else:
            z_random = self.get_z_random(real_image.size(0), self.nz)
        _, pos, negs = self.latent_augmented_sampling(z_random)
        z_conc = [z_random, pos] + negs
        z_conc = torch.cat(z_conc,dim=0)
        label_conc = torch.cat([label] * (self.opt.num_negative + 2), 0)
        z_r, down_outputs = self.E.forward(ref_image)
        z_r_conc = torch.cat([z_r] * (self.opt.num_negative + 2), 0)
        down_outputs_conc = [x.repeat(self.opt.num_negative + 2, 1, 1, 1) for x in down_outputs]

        fake_image = self.G.forward(down_outputs_conc, z_conc, z_r_conc, label_conc)
        outputs = fake_image[:real_image.size(0)]
        fake_image_rs = torch.cat((ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256), \
                                   outputs, ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)), dim=1)
        pred_fake, fake_label = self.D.forward(fake_image_rs)

        if enLoss:
            ## D
            real_image_rs = torch.cat((ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256), \
                                       real_image.view(self.opt.batch_size, 1, 256, 256) \
                                           , ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)), dim=1)
            pred_real, real_label = self.D.forward(real_image_rs)

            all1_real = torch.ones_like(pred_real).cuda()
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(pred_real, all1_real)
            all0_fake = torch.zeros_like(pred_fake).cuda()
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all0_fake)

            loss_D_fake = ad_true_loss + ad_fake_loss
            self.loss_val_D_fake_GAN = ad_fake_loss
            self.loss_val_D_real_GAN = ad_true_loss

            self.loss_val_D_GAN = loss_D_fake
            tarlabel = torch.zeros_like(label)
            tarlabel = tarlabel.cuda()
            self.loss_val_D_label = self.compute_mse_loss(fake_label, tarlabel) + \
                                    self.compute_mse_loss(real_label, label)
            self.loss_val_D = self.loss_val_D_label + self.loss_val_D_GAN

            ## G
            ref1 = ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256)
            ref1s = ref1.repeat(self.opt.num_negative + 2, 1, 1, 1)
            ref2 = ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)
            ref2s = ref2.repeat(self.opt.num_negative + 2, 1, 1, 1)
            ref_fake = torch.cat((ref1s, fake_image, ref2s), dim=1)
            _, _,feats_real = self.D.forward(ref_fake, enc_feat=True)

            all_ones_fake = torch.ones_like(pred_fake).cuda()
            advloss = nn.functional.binary_cross_entropy_with_logits(pred_fake, all_ones_fake)
            labelloss = self.compute_mse_loss(fake_label, label)

            self.loss_val_G_GAN = advloss
            # content loss
            self.loss_val_content = self.compute_mse_loss(outputs.view(self.opt.batch_size, 256, 256),real_image)
            # label loss
            self.loss_val_G_label = labelloss
            # contrastive loss
            self.loss_val_contra = 0.0
            for i in range(real_image.size(0)):
                logits_real = feats_real[i:feats_real.shape[0]:real_image.size(0)].view(self.opt.num_negative + 2,
                                                                                           -1)
                if self.opt.featnorm:
                    logits_real = logits_real / torch.norm(logits_real, p=2, dim=1, keepdim=True)
                contra_real = self.compute_contrastive_loss(logits_real[0:1], logits_real[1:])
                contra = contra_real
                self.loss_val_contra += contra

            self.loss_val_G = self.loss_val_G_GAN * self.opt.gamma_genL1 + self.loss_val_contra * self.opt.gamma_genContra + \
                          self.loss_val_G_label * self.opt.gamma_genLabel + self.loss_val_content * self.opt.gamma_genMSE

        return outputs.permute(0, 2, 3, 1).cpu().detach().numpy(), fake_label.cpu().detach().numpy()

    def test_forward_(self, image, label,sample_z=0,inputz=False,en_fea=False):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_image = torch.from_numpy(image[:, :, :, (0, 2)]).permute(0, 3, 1, 2).cuda()
        label = torch.Tensor(label).cuda()
        real_image = torch.from_numpy(image[:,:,:,1]).cuda()
        # rand z
        if inputz:
            z_random = sample_z.cuda()
        else:
            z_random = self.get_z_random(real_image.size(0), self.nz).cuda()
        z_r, down_outputs = self.E.forward(ref_image)
        fake_image = self.G.forward(down_outputs, z_random, z_r, label)
        outputs = fake_image[:real_image.size(0)]
        fake_image_rs = torch.cat((ref_image[:, 0, :, :].view(self.opt.batch_size, 1, 256, 256), \
                                   outputs, ref_image[:, 1, :, :].view(self.opt.batch_size, 1, 256, 256)), dim=1)
        if en_fea:
            pred_fake, fake_label, feats_real = self.D.forward(fake_image_rs,enc_feat=en_fea)
            return outputs.permute(0, 2, 3, 1).cpu().detach().numpy(), fake_label.cpu().detach().numpy(), feats_real.cpu().detach().numpy()
        else:
            pred_fake, fake_label = self.D.forward(fake_image_rs,enc_feat=en_fea)
            return outputs.permute(0, 2, 3, 1).cpu().detach().numpy(), fake_label.cpu().detach().numpy()

