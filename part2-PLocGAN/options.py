import argparse
import os
from config import *

workdirec = './'
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.namestr = '_cl_CYNP'
        self.parser.add_argument('--logs', type=str, default=self.namestr, help='current work directory')
        self.parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')

        self.parser.add_argument('--working_directory', type=str, default=workdirec, help='current work directory')
        self.parser.add_argument('--modeldir', type=str, default=workdirec+'modeldir_unet_guide' + self.namestr, help='The stored model dir ')
        self.parser.add_argument('--sampledir', type=str, default=workdirec+'sampledir_unet_guide' + self.namestr, help='Many others stored dir ')
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

        self.parser.add_argument('--checkpoint', type=int, default=40, help='number of epochs to be reloaded')
        #
        self.parser.add_argument('--n_class', type=int, default=2, help='number of classes')
        self.parser.add_argument('--height', type=int, default=IMGSIZE, help='height of image')
        self. parser.add_argument('--width', type=int, default=IMGSIZE, help='width of image')
        self.parser.add_argument('--gan_noise', type=float, default=0.01, help='injection noise for the GAN')
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size')  # ref:10; 5
        self.parser.add_argument('--hiddenz_size', type=int, default=16, help='size of the hidden z')
        self.parser.add_argument('--hiddenr_size', type=int, default=8, help='size of the hidden r')
        self.parser.add_argument('--noise_bool', action='store_true', default=False,help='add noise on all GAN layers or not')

        # DivCo related
        self.parser.add_argument('--featnorm', action='store_true', help='whether featnorm')
        self.parser.add_argument('--radius', type=float, default=0.01, help='positive sample - distance threshold')
        self.parser.add_argument('--tau', type=float, default=1.0, help='temperature')
        self.parser.add_argument('--num_negative', type=int, default=5,help='number of latent negative samples')  # ref:10; 5



class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--logsname', type=str, default='trainlogs', help='folder name to save outputs')
        self.parser.add_argument('--logs_val_name', type=str, default='vallogs', help='folder name to save outputs')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--resume', type=str, default=None,help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')
        # Dir
        self.parser.add_argument('--logdir', type=str, default=workdirec+'logdir_unet_guide' + self.namestr, help='The stored train log dir ')
        self.parser.add_argument('--ImgsVerifydir', type=str, default=workdirec+'imgs_unet_guide' + self.namestr,help='Stored dir for Valid Generate Images')
        self.parser.add_argument('--train_Metric', type=str, default= os.path.join(workdirec+'sampledir_unet_guide' + self.namestr,\
                                                                                   'train_metric' + self.namestr + '.csv'),\
                                 help='Stored evaluated txt for training')
        self.parser.add_argument('--valid_Metric', type=str,
                                 default=os.path.join(workdirec+'sampledir_unet_guide' + self.namestr, \
                                                      'valid_metric' + self.namestr + '.csv'), \
                                 help='Stored evaluated txt for validing')
        self.parser.add_argument('--max_epoch', type=int, default=50,
                            help='max epoch for total training')
        self.parser.add_argument('--save_step', type=int, default=3,
                                 help='save model per #save_step epoch')
        # gamma parameters
        self.parser.add_argument('--gamma_genMSE', type=float, default=1, help='Content Loss for Generator')
        self.parser.add_argument('--gamma_genL1', type=float, default=0.1, help='Adversarial Loss for Generator')
        self.parser.add_argument('--gamma_genLabel', type=float, default=1, help='Label Loss for Generator')
        self.parser.add_argument('--gamma_genContra', type=float, default=0.01, help='Contrastive Loss for Generator') # 1no; 0.1no; 0.01; 0.03

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt