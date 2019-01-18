import torch
import torch.nn as nn
import os
from layers import *
from .networks import vgg, vgg_base, BasicConv, BasicRFB, BasicRFB_a

class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.
    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, bn=True):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.backbone = nn.ModuleList(base)
        # conv_4
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0, bn=bn)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.backbone[k](x)
        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.backbone)):
            x = self.backbone[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(size, cfg, i, bn=True):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=1, bn=bn)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2, bn=bn)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2, bn=bn)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1,bn=bn)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1,bn=bn)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1,bn=bn)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1,bn=bn)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1,bn=bn)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1,bn=bn)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    # indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21, bn=True):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(vgg_base[str(size)], 3),
                                         add_extras(size, extras[str(size)], 1024, bn=bn),
                                         mbox[str(size)], num_classes), num_classes, bn=bn)
