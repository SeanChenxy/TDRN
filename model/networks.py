import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from utils._ext import deform_conv
import math
import cv2

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512], # output channel
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    'VOC_300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'MOT_300': [5, 5, 5, 5, 5, 5],
    'VOC_512': [6, 6, 6, 6, 6, 4, 4],
}

def xavier(param):
    init.xavier_uniform_(param)

def orthogonal(param):
    init.orthogonal_(param)

def zeros_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.zero_()
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d):
        orthogonal(m.weight.data)
        m.bias.data.fill_(1)

def rfb_weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

def net_init(ssd_net, backbone, logging, attention=False, pm=0.0, refine=False, deform=False, multihead=False):
    # initialize newly added layers' weights with xavier method
    if 'RFB' in backbone:
        logging.info('Initializing extra, Norm, loc, conf ...')
        ssd_net.extras.apply(rfb_weights_init)
        ssd_net.loc.apply(rfb_weights_init)
        ssd_net.conf.apply(rfb_weights_init)
        ssd_net.Norm.apply(rfb_weights_init)
        if 'E_VGG' in backbone:
            logging.info('Initializing reduce, up_reduce ...')
            ssd_net.reduce.apply(rfb_weights_init)
            ssd_net.up_reduce.apply(rfb_weights_init)
    elif 'RefineDet' in backbone:
        if 'ResNet' in backbone:
            logging.info('Initializing post_up_layers, upsample_latent_layers ...')
            ssd_net.post_up_layers.apply(weights_init)
            ssd_net.upsample_latent_layers.apply(weights_init)
        else:
            logging.info('Initializing extras, last_layer_trans ...')
            ssd_net.extras.apply(weights_init)
            ssd_net.last_layer_trans.apply(weights_init)
        logging.info('Initializing, trans_layers, latent_layers, up_layers ...')
        ssd_net.trans_layers.apply(weights_init)
        ssd_net.latent_layers.apply(weights_init)
        ssd_net.up_layers.apply(weights_init)
        if not deform:
            logging.info( 'Initializing odm_loc, odm_conf ...')
            ssd_net.odm_loc.apply(weights_init)
            ssd_net.odm_conf.apply(weights_init)
            if multihead:
                logging.info('Initializing odm_loc2, odm_conf2 ...')
                ssd_net.odm_loc_2.apply(weights_init)
                ssd_net.odm_conf_2.apply(weights_init)
        else:
            logging.info('Initializing offset ...')
            ssd_net.offset.apply(weights_init)
            if multihead:
                logging.info('Initializing offset2 ...')
                ssd_net.offset2.apply(weights_init)
        if refine:
            logging.info('Initializing arm_loc...')
            ssd_net.arm_loc.apply(weights_init)
    else:
        logging.info('Initializing extras ...')
        ssd_net.extras.apply(weights_init)
        if '4s' in backbone:
            if not deform:
                logging.info('Initializing arm_loc, arm_conf ...')
                ssd_net.arm_loc.apply(weights_init)
                ssd_net.arm_conf.apply(weights_init)
            else:
                logging.info('Initializing offset ...')
                ssd_net.offset.apply(weights_init)
        else:
            logging.info('Initializing loc, conf ...')
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
        if pm != 0.0:
            ssd_net.pm.apply(weights_init)
        if attention:
            logging.info('Initializing Attention weights...')
            ssd_net.attention.apply(weights_init)

########### VGGNet #############
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False, pool5_ds=False, c7_channel=1024):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if pool5_ds:
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    else:
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    conv7 = nn.Conv2d(1024, c7_channel, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(conv6.out_channels),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(conv7.out_channels), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1), nn.BatchNorm2d(cfg[k + 1])]
                else:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                if batch_norm and k in [7]:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]), nn.BatchNorm2d(v)]

                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


class ConvAttention(nn.Module):

    def __init__(self, inchannel):
        super(ConvAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(inchannel, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feats):
        return self.attention(feats)
########### ConvRNN #############
# https://www.jianshu.com/p/72124b007f7d
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
                torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
            )

        prev_cell, prev_hidden = prev_state
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return cell, hidden

    def init_state(self, input_):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
        )
        return state

class ConvJANET(nn.Module):
    """
    Generate a convolutional JANET cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvJANET, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),)

        prev_cell = prev_state[-1]
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_cell), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        remember_gate, cell_gate = gates.chunk(2, 1)

        # apply sigmoid non linearity
        remember_gate = F.sigmoid(remember_gate)
        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + ((1-remember_gate) * cell_gate)

        return (cell, )

    def init_state(self, input_):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),)
        return state

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, cuda_flag=True, phase='train'):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)
        self.phase = phase

    def forward(self, input, pre_state):
        if pre_state is None:
            size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
            pre_state = (torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test']).cuda(),)

        hidden = pre_state[-1]
        c1 = self.ConvGates(torch.cat((F.dropout(input,p=0.2,training=(False,True)[self.phase=='train']), hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = F.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return (next_h, )

    def init_state(self, input):
        size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
        state = torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test']).cuda(),
        return state

########### ResNet #############
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

########### RFB Model #############
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1, bn=True):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride, bn=bn),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False, bn=bn)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), bn=bn),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False, bn=bn)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, bn=bn),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, bn=bn),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False, bn=bn)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False, bn=bn)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, bn=bn)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_c(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1, bn=True):
        super(BasicRFB_c, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            # BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, bn=bn),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False, bn=bn)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, bn=bn),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, bn=bn),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False, bn=bn)
        )
        self.branch4 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, bn=bn),
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False, bn=bn)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, bn=bn)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x4 = self.branch4(x)

        out = torch.cat((x0 ,x1, x4), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, bn=True):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False, bn=bn)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0), bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, bn=bn)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1), bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False, bn=bn)
        )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1, bn=bn),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bn=bn),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0), bn=bn),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False, bn=bn)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False, bn=bn)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False, bn=bn)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

########### Prediction Model #############
class PreModule(nn.Module):

    def __init__(self, inchannl, channel_increment_factor):
        super(PreModule, self).__init__()
        self.inchannel = inchannl
        self.pm = nn.Sequential(
            nn.Conv2d(inchannl, inchannl, kernel_size=1),
            nn.Conv2d(inchannl, inchannl, kernel_size=1),
            nn.Conv2d(inchannl, int(inchannl*channel_increment_factor), kernel_size=1)
        )
        self.extend = nn.Conv2d(inchannl, int(inchannl*channel_increment_factor), kernel_size=1)

    def forward(self, x):
        return self.extend(x) + self.pm(x)

########### Deformable Conv #############
def conv_offset2d(input,
                  offset,
                  weight,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deform_groups=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = ConvOffset2dFunction(
        _pair(stride), _pair(padding), _pair(dilation), deform_groups)
    return f(input, offset, weight)

class ConvOffset2dFunction(Function):
    def __init__(self, stride, padding, dilation, deformable_groups=1):
        super(ConvOffset2dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)

        output = input.new(*self._output_size(input, weight))

        self.bufs_ = [input.new(), input.new()]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            # if isinstance(input, torch.autograd.Variable):
            #     if not isinstance(input.data, torch.cuda.FloatTensor):
            #         raise NotImplementedError
            # else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            deform_conv.deform_conv_forward_cuda(
                input, weight, offset, output, self.bufs_[0], self.bufs_[1],
                weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                self.padding[1], self.padding[0], self.dilation[1],
                self.dilation[0], self.deformable_groups)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            # if isinstance(grad_output, torch.autograd.Variable):
            #     if not isinstance(grad_output.data, torch.cuda.FloatTensor):
            #         raise NotImplementedError
            # else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                deform_conv.deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, self.bufs_[0], weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0], self.deformable_groups)

            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv.deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, self.bufs_[0], self.bufs_[1], weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0], self.deformable_groups, 1)

        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

class ConvOffset2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1):
        super(ConvOffset2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        # stdv = 1. / math.sqrt(n)
        xavier(self.weight.data)
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return conv_offset2d(input, offset, self.weight, self.stride,
                             self.padding, self.dilation,
                             self.num_deformable_groups)

########### MobileNet #############
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def MobileNet():
    layers = []
    layers += [nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                             nn.BatchNorm2d(32),
                             nn.ReLU(inplace=True))]
    layers += [conv_dw(32, 64, 1)]
    layers += [conv_dw(64, 128, 2)]
    layers += [conv_dw(128, 128, 1)]
    layers += [conv_dw(128, 256, 2)] # 40
    layers += [conv_dw(256, 256, 1)]
    layers += [conv_dw(256, 512, 2)] # 20
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 1024, 2)] # 10
    layers += [conv_dw(1024, 1024, 1)]

    return layers

########### SqueezeNet #############
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        self.features = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2), # 160
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 80
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 40
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),  # p1
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 20
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),  # p2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 10
            Fire(512, 80, 384, 384),
            Fire(768, 80, 384, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 5
            Fire(768, 96, 448, 448),
            Fire(896, 96, 448, 448),
        ]
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
