import torch
import torch.nn as nn

####################################################################
#------------------------- Generator -------------------------------
####################################################################
class generator(nn.Module):
    # initializers
    def __init__(self, opts):
        super(generator, self).__init__()
        self.opt = opts
        self.fc1 = nn.Linear(opts.hiddenr_size+opts.hiddenz_size+opts.n_class,1024*4*4,bias=False)
        self.fc1_bn = nn.BatchNorm1d(1024*4*4,eps=0.001,momentum=0.001)
        self.prelu1 = nn.PReLU(init=0.0)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.gated_deconv1 = gated_deconv(opts, in_ch=1024, out_ch=1024, size=4)
        self.gated_deconv2 = gated_deconv(opts, in_ch=1024, out_ch=512, size=8)
        self.gated_deconv3 = gated_deconv(opts, in_ch=512, out_ch=256, size=16)
        self.gated_deconv4 = gated_deconv(opts, in_ch=256, out_ch=128, size=32)
        self.gated_deconv5 = gated_deconv(opts, in_ch=128, out_ch=64, size=64)

        self.fc2 = nn.Linear(opts.n_class, 128 * 128)
        self.sigmoid1 = nn.Sigmoid()
        self.deconv1 = nn.Conv2d(128, 1, 4, 1, padding=0)
        self.upsample = nn.Upsample(size=[128 * 2 - 2, 128 * 2 - 2], mode='nearest')
        self.pad = nn.ZeroPad2d(padding=(2, 3, 2, 3))

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self,down_outputs, z_s, z_r, y):
        cond = torch.cat([z_r, y], 1)
        z = torch.cat([cond, z_s], 1)
        output = self.fc1_bn(self.fc1(z))
        output = torch.reshape(output, [-1, 1024, 4, 4])
        output = self.prelu1(output)
        output = self.gated_deconv1(inputs=output, skip_input=down_outputs[5], y=y, size=4)
        output = self.gated_deconv2(inputs=output, skip_input=down_outputs[4], y=y, size=8)
        output = self.gated_deconv3(inputs=output, skip_input=down_outputs[3], y=y, size=16)
        output = self.gated_deconv4(inputs=output, skip_input=down_outputs[2], y=y, size=32)
        output = self.gated_deconv5(inputs=output, skip_input=down_outputs[1], y=y, size=64)

        prob_map = self.sigmoid1(self.fc2(y))
        prob_map = torch.reshape(prob_map, [-1, 1, 128, 128])
        skip_out = torch.mul(down_outputs[0], prob_map)
        output = torch.cat([output, skip_out], 1)
        output = self.deconv1(self.pad(self.upsample(output)))
        output = self.sigmoid2(output)

        return output

####################################################################
# ------------------------ Encoder ---------------------------------
####################################################################
class Encoder(nn.Module):
    def __init__(self,opts,d=64):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1,d,4,2,1,bias=False)
        self.conv1_bn = nn.BatchNorm2d(d,eps=0.001,momentum=0.001)
        self.prelu1 = nn.PReLU(init=0.0)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1,bias=False)
        self.conv2_bn = nn.BatchNorm2d(d*2,eps=0.001,momentum=0.001)
        self.prelu2 = nn.PReLU(init=0.0)

        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1,bias=False)
        self.conv3_bn = nn.BatchNorm2d(d*4,eps=0.001,momentum=0.001)
        self.prelu3 = nn.PReLU(init=0.0)

        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1,bias=False)
        self.conv4_bn = nn.BatchNorm2d(d*8,eps=0.001,momentum=0.001)
        self.prelu4 = nn.PReLU(init=0.0)

        self.conv5 = nn.Conv2d(d*8, d*16, 4, 2, 1,bias=False)
        self.conv5_bn = nn.BatchNorm2d(d*16,eps=0.001,momentum=0.001)
        self.prelu5 = nn.PReLU(init=0.0)

        self.conv6 = nn.Conv2d(d*16, d*16, 4, 2, 1,bias=False)
        self.conv6_bn = nn.BatchNorm2d(d*16,eps=0.001,momentum=0.001)

        self.prelu6 = nn.PReLU(init=0.0)

        self.fc1 = nn.Linear(4*4*d*16,opts.hiddenr_size,bias=False)
        self.fc1_bn = nn.BatchNorm1d(opts.hiddenr_size,eps=0.001,momentum=0.001)

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])


    def forward(self, input,d=64):
        down_outputs = []
        x = self.prelu1(self.conv1_bn(self.conv1(input)))
        down_outputs.append(x)
        x = self.prelu2(self.conv2_bn(self.conv2(x)))
        down_outputs.append(x)
        x = self.prelu3(self.conv3_bn(self.conv3(x)))
        down_outputs.append(x)
        x = self.prelu4(self.conv4_bn(self.conv4(x)))
        down_outputs.append(x)
        x = self.prelu5(self.conv5_bn(self.conv5(x)))
        down_outputs.append(x)
        x = self.conv6_bn(self.conv6(x))
        down_outputs.append(x)
        x = x.view(-1,4*4*d*16)
        x = self.prelu6(x)
        y = self.fc1_bn(self.fc1(x))
        return y,down_outputs

####################################################################
#------------------------- gated_deconv ----------------------------
####################################################################
class gated_deconv(nn.Module):
    def __init__(self,opts,in_ch, out_ch, size):
        super(gated_deconv, self).__init__()
        self.gate_fc = nn.Sequential(nn.Linear(opts.n_class, size * size),nn.Sigmoid())

        ##
        self.Upsample = nn.Upsample(size=[size*2-2,size*2-2], mode='nearest')
        self.pad = nn.ZeroPad2d(padding=(2, 3, 2, 3))
        self.gate_deconv = nn.Sequential(nn.Conv2d(in_ch*2, out_ch, 4, 1, padding=0,bias=False), \
                                         nn.BatchNorm2d(out_ch,eps=0.001,momentum=0.001),\
                                         nn.PReLU(init=0.0))
        model = []
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.gate_fc.apply(gaussian_weights_init)
        self.gate_deconv.apply(gaussian_weights_init)

    def forward(self, inputs, skip_input, y, size):
        prob_map = self.gate_fc(y)
        prob_map = torch.reshape(prob_map, [-1, 1, size, size])
        output = torch.mul(skip_input, prob_map)
        output = torch.cat([inputs, output], 1)
        output = self.pad(self.Upsample(output))
        output = self.gate_deconv(output)

        return output

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class discriminator(nn.Module):
    # initializers
    def __init__(self, ndf=32):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(2,ndf,4,2,1,bias=False)
        self.conv1_bn = nn.BatchNorm2d(ndf,eps=0.001,momentum=0.001)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1,bias=False)
        self.conv2_bn = nn.BatchNorm2d(ndf*2,eps=0.001,momentum=0.001)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1,bias=False)
        self.conv3_bn = nn.BatchNorm2d(ndf * 4,eps=0.001,momentum=0.001)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1,bias=False)
        self.conv4_bn = nn.BatchNorm2d(ndf * 8,eps=0.001,momentum=0.001)
        self.lrelu4 = nn.LeakyReLU(0.2)


        self.fc1 = nn.Linear(16*16*ndf*8,256,bias=False)
        self.fc1_bn = nn.BatchNorm1d(256,eps=0.001,momentum=0.001)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(256,1)

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, input_X,ndf=32,gan_noise=0.01):

        output = input_X
        if gan_noise > 0:
            output = add_white_noise(output)

        output = self.lrelu1(self.conv1_bn(self.conv1(output)))
        output = self.lrelu2(self.conv2_bn(self.conv2(output)))
        output = self.lrelu3(self.conv3_bn(self.conv3(output)))
        output = self.lrelu4(self.conv4_bn(self.conv4(output)))
        output = output.view(-1, 16*16*ndf*8)
        output = self.sigmoid(self.fc1_bn(self.fc1(output)))
        Doutput = self.fc2(output)
        return Doutput

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


def conv_cond_concat(x, y):
    x_shapes = x.shape
    y_shapes = y.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep_y = torch.ones(y_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]).to(device)
    return torch.cat((
        x, y * rep_y), dim=1)

def add_white_noise(input_tensor, mean=0, stddev=0.01):
    input_shape = input_tensor.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.normal(mean, stddev,(input_shape[0], input_shape[1], input_shape[2], input_shape[3])).to(device)
    return input_tensor+noise
