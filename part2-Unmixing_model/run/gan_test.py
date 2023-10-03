import argparse
import os
from run.gan_model import DivCo_PLGAN
from utils.common_util import *
from config.config_ import *
from datasets.tool import image_to_tensor
from scipy.io import loadmat

gan_modelnamelist = {'CyNp': 'CyNp.pth',
                     'CyPM': 'CyPM.pth',
                     'NuMi': 'NuMi.pth',
                     'NuNu': 'NuNu.pth',
                     'LyMo': 'LyMo.pth'}
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--working_directory', type=str, default='../../', help='current work directory')
        #
        self.parser.add_argument('--n_class', type=int, default=2, help='number of classes')
        self.parser.add_argument('--height', type=int, default=256, help='height of image')
        self. parser.add_argument('--width', type=int, default=256, help='width of image')
        self.parser.add_argument('--hiddenz_size', type=int, default=16, help='size of the hidden z')
        self.parser.add_argument('--hiddenr_size', type=int, default=8, help='size of the hidden r')

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt


class ganmodel_():
    def __init__(self):
        super(ganmodel_, self).__init__()
        parser = TestOptions()
        self.opts = parser.parse()
        self.model_cynp = DivCo_PLGAN(self.opts,comb='cynp')
        self.model_cynp.eval()
        self.model_cynp.setgpu()
        model_dir_cynp = opj(self.opts.working_directory, gan_modelnamelist['CyNp'])
        print('Load Trained %s.pth' %('CyNp'))
        self.model_cynp.resume(model_dir_cynp)

        self.model_cypm = DivCo_PLGAN(self.opts,comb='cypm')
        self.model_cypm.eval()
        self.model_cypm.setgpu()
        model_dir_cypm = opj(self.opts.working_directory,  gan_modelnamelist['CyPM'])
        print('Load Trained %s.pth' %('CyPM'))
        self.model_cypm.resume(model_dir_cypm)

        self.model_numi = DivCo_PLGAN(self.opts,comb='numi')
        self.model_numi.eval()
        self.model_numi.setgpu()
        model_dir_numi = opj(self.opts.working_directory, gan_modelnamelist['NuMi'])
        print('Load Trained %s.pth' %('NuMi'))
        self.model_numi.resume(model_dir_numi)

        self.model_nunu = DivCo_PLGAN(self.opts,comb='nunu')
        self.model_nunu.eval()
        self.model_nunu.setgpu()
        model_dir_nunu = opj(self.opts.working_directory,  gan_modelnamelist['NuNu'])
        print('Load Trained %s.pth' %('NuNu'))
        self.model_nunu.resume(model_dir_nunu)

        self.model_lymo = DivCo_PLGAN(self.opts, comb='lymo')
        model_dir_lymo = opj(self.opts.working_directory, gan_modelnamelist['LyMo'])
        print('Load Trained %s.pth' % ('LyMo'))
        self.model_lymo.resume(model_dir_lymo)


    def gan_test_(self,img,label_5,label_2,comb):
        if comb == 'NuMi'or comb == 'Mi':
            self.model = self.model_numi
        elif comb == 'NuNu'or comb == 'Nu':
            self.model = self.model_nunu
        elif comb == 'CyNp' or comb == 'Np' or comb == 'Cy':
            self.model = self.model_cynp
        elif comb == 'CyPM' or comb == 'PM':
            self.model = self.model_cypm
        elif comb == 'lymo':
            self.model = self.model_lymo
        elif comb == 'None':
            raise ValueError('comb can not be None!')

        if not dataset == 'lymo':
            output = self.model.test_forward(img.reshape([-1,256,256,3]), label_2)
            output = output.reshape([256,256,3])
        elif dataset == 'lymo':
            output = self.model.test_forward(img.reshape([-1, 2, 256, 256]), label_2)
            output = output.reshape([256, 256, 2])
        return output





