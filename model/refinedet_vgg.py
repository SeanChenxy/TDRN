import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from .networks import vgg, vgg_base, BasicConv

class RefineSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self,size, num_classes=21, use_refine=False, phase='train', c7_channel=1024, bn=False, multihead=False):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.use_refine = use_refine
        self.phase = phase
        self.bn = bn
        self.multihead = multihead
        num_box = 3

        # SSD network
        self.backbone = nn.ModuleList(vgg(vgg_base['320'], 3, batch_norm=self.bn, pool5_ds=True, c7_channel=c7_channel))
        self.conv4_3_layer = (23, 33)[self.bn]
        self.conv5_3_layer = (30, 43)[self.bn]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        if self.bn:
            self.extras = nn.Sequential(nn.Conv2d(c7_channel, 256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))
        else:
            self.extras = nn.Sequential(nn.Conv2d(c7_channel, 256, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        if use_refine:
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(c7_channel, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          ])
            # self.arm_conf = nn.ModuleList([nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1), \
            #                                nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1), \
            #                                nn.Conv2d(1024, num_box*self.num_classes, kernel_size=3, stride=1, padding=1), \
            #                                nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1), \
            #                                ])
        self.odm_loc = nn.ModuleList([nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1),
                                      ])
        self.odm_conf = nn.ModuleList([nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       ])
        if self.multihead:
            self.odm_loc_2 = nn.ModuleList([nn.Conv2d(256, num_box * 4, kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(256, num_box * 4, kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(256, num_box * 4, kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(256, num_box * 4, kernel_size=5, stride=1, padding=2),
                                          ])
            self.odm_conf_2 = nn.ModuleList(
                [nn.Conv2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2),
                 nn.Conv2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2),
                 nn.Conv2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2),
                 nn.Conv2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2),
                 ])
        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           nn.Sequential(nn.Conv2d(c7_channel, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           ])

        self.up_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0), ])
        self.latent_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           ])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()
        arm_loc_list = list()
        # arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(self.conv4_3_layer):
            x = self.backbone[k](x)

        s = self.L2Norm_4_3(x)
        arm_sources.append(s)

        # apply vgg up to conv5_3
        for k in range(self.conv4_3_layer , self.conv5_3_layer ):
            x = self.backbone[k](x)
        s = self.L2Norm_5_3(x)
        arm_sources.append(s)

        # apply vgg up to fc7
        for k in range(self.conv5_3_layer , len(self.backbone)):
            x = self.backbone[k](x)
        arm_sources.append(x)
        # conv6_2
        x = self.extras(x)
        arm_sources.append(x)
        # apply multibox head to arm branch
        if self.use_refine:
            for (a, l) in zip(arm_sources, self.arm_loc):
                arm_loc_list.append(l(a).permute(0, 2, 3, 1).contiguous())
                # arm_conf_list.append(c(a).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            # arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
        x = self.last_layer_trans(x)
        obm_sources.append(x)

        # get transformed layers
        trans_layer_list = list()
        for (x_t, t) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t(x_t))
        # fpn module
        trans_layer_list.reverse()
        arm_sources.reverse()
        for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layers):
            x = F.relu(l(F.relu(u(x) + t, inplace=True)), inplace=True)
            obm_sources.append(x)
        obm_sources.reverse()
        if self.multihead:
            for (ob, l, c, l2, c2) in zip(obm_sources, self.odm_loc, self.odm_conf, self.odm_loc_2, self.odm_conf_2):
                obm_loc_list.append((l(ob)+l2(ob)).permute(0, 2, 3, 1).contiguous())
                obm_conf_list.append((c(ob)+c2(ob)).permute(0, 2, 3, 1).contiguous())
        else:
            for (ob, l, c) in zip(obm_sources, self.odm_loc, self.odm_conf):
                obm_loc_list.append(l(ob).permute(0, 2, 3, 1).contiguous())
                obm_conf_list.append(c(ob).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)

        # apply multibox head to source layers

        if self.phase == 'test':
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    None, #self.softmax(arm_conf.view(-1, self.num_classes)),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
        else:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    None, # arm_conf.view(arm_conf.size(0), -1, self.num_classes),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_net(phase, size=320, num_classes=21, use_refine=False, c7_channel=1024, bn=False, multihead=False):
    if size not in [320, 512]:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return RefineSSD(size, num_classes=num_classes, use_refine=use_refine, phase=phase, c7_channel=c7_channel, bn=bn, multihead=multihead)
