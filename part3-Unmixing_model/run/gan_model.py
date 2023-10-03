import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
import networks.gan as gan
from torch.nn import DataParallel
from config.config_ import *

class DivCo_PLGAN(nn.Module):
    def __init__(self, opts,comb):
        super(DivCo_PLGAN, self).__init__()
        self.nz = opts.hiddenz_size
        self.opt = opts
        self.class_num = opts.n_class
        self.E = gan.Encoder(opts,comb) ###
        self.G = gan.generator(opts)  ###
        self.comb = comb


    def setgpu(self):
        # self.E = DataParallel(self.E).cuda()
        # self.G = DataParallel(self.G).cuda()
        self.E.cuda()
        self.G.cuda()

    def get_z_random(self, batchSize, nz):
        z = torch.cuda.FloatTensor(batchSize, nz)
        if self.comb == 'cynp':
            zmax = -0.45
            zmin=-1.0
        elif self.comb == 'cypm':
            zmax = 0.7
            zmin = 0
        elif self.comb == 'numi':
            zmax = 1.0
            zmin = -0.35
        elif self.comb == 'nunu':
            zmax = -0.1
            zmin = -1.0
        elif self.comb == 'lymo':
            zmax = 1.0
            zmin = -0.5
        else:
            zmax = 1.0
            zmin = -1.0
        z.copy_(torch.clamp(torch.randn(batchSize, nz),zmin,zmax))
        return z

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir) ###
        self.E.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['enc'].items()})
        self.G.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['gen'].items()})
        return checkpoint['ep'], checkpoint['total_it']

    def test_forward(self, image, label):
        if not dataset == 'lymo':
            ref_image = torch.from_numpy(image[:, :, :, (0, 2)]).permute(0, 3, 1, 2).cuda()
            label = torch.Tensor(label).cuda()
            z_random = self.get_z_random(ref_image.size(0), self.nz).cuda()
            z_r, down_outputs = self.E.forward(ref_image)
            outputs = self.G.forward(down_outputs, z_random, z_r, label)
            outputs = torch.cat((ref_image[:, 0, :, :].view(-1, 1, 256, 256), outputs, ref_image[:, 1, :, :].view(-1, 1, 256, 256)), dim=1)
            return outputs.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            ref_image = torch.from_numpy(image[:, :, :, 0].view(-1, 256, 256,1)).permute(0, 3, 1, 2).cuda()
            label = torch.Tensor(label).cuda()
            z_random = self.get_z_random(ref_image.size(0), self.nz).cuda()
            z_r, down_outputs = self.E.forward(ref_image)
            outputs = self.G.forward(down_outputs, z_random, z_r, label)
            outputs = torch.cat((ref_image, outputs), dim=1)
            return outputs.permute(0, 2, 3, 1).cpu().detach().numpy()



