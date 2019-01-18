import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from .networks import conv_dw

class RefineSSD(nn.Module):

    def __init__(self,size, num_classes=21, use_refine=False, phase='train'):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.use_refine = use_refine
        self.phase = phase
        num_box = 3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(1024, 8)
        self.backbone = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                          nn.BatchNorm2d(32),
                          nn.ReLU(inplace=True)),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),  # 40 original stride=2
            conv_dw(256, 256, 1),  # p1
            conv_dw(256, 512, 2),  # 20
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),  # p2
            conv_dw(512, 1024, 2),  # 10
            conv_dw(1024, 1024, 1),  # p3
        ])
        self.extras = nn.ModuleList([nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.BatchNorm2d(256),
                                                   conv_dw(256, 512, 2)),
                                     nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.BatchNorm2d(256),
                                                   conv_dw(256, 512, 2))
                                     ])
        self.last_layer_trans = nn.Sequential(conv_dw(512, 256, 1), #nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              # nn.ReLU(inplace=True),
                                              conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              conv_dw(256, 256, 1, relu=False),) #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        if use_refine:
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.Conv2d(1024, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                          ])

        self.odm_loc = nn.ModuleList([
                                      # conv_dw(256, num_box * 4, 1, bn=False, relu=False),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      # conv_dw(256, num_box * 4, 1, bn=False, relu=False),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      # conv_dw(256, num_box * 4, 1, bn=False, relu=False),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      # conv_dw(256, num_box * 4, 1, bn=False, relu=False),
                                      nn.Conv2d(256, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      ])
        self.odm_conf = nn.ModuleList([
                                       # conv_dw(256, num_box * self.num_classes, 1, bn=False, relu=False),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       # conv_dw(256, num_box * self.num_classes, 1, bn=False, relu=False),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       # conv_dw(256, num_box * self.num_classes, 1, bn=False, relu=False),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       # conv_dw(256, num_box * self.num_classes, 1, bn=False, relu=False),
                                       nn.Conv2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       ])
        self.trans_layers = nn.ModuleList([nn.Sequential(conv_dw(512, 256, 1), #nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         #nn.ReLU(inplace=True),
                                                         conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
                                                         ),
                                           nn.Sequential(conv_dw(1024, 256, 1), #nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         #nn.ReLU(inplace=True),
                                                         conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
                                                         ),
                                           nn.Sequential(conv_dw(512, 256, 1,), #nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         #n.ReLU(inplace=True),
                                                         conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
                                                         ),
                                           ])

        self.up_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False), ])
        self.latent_layers = nn.ModuleList([conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            conv_dw(256, 256, 1, relu=False), #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           ])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        arm_sources = list()
        arm_loc_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()

        for i in range(12):
            x = self.backbone[i](x)
        arm_sources.append(self.L2Norm_4_3 (x))
        for i in range(12, len(self.backbone)):
            x = self.backbone[i](x)
        arm_sources.append(self.L2Norm_5_3 (x))

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            arm_sources.append(x)

        # apply multibox head to arm branch
        if self.use_refine:
            for (a, l) in zip(arm_sources, self.arm_loc):
                arm_loc_list.append(l(a).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
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

def build_net(phase, size=320, num_classes=21, use_refine=False):
    if size not in [320, 512]:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return RefineSSD(size, num_classes=num_classes, use_refine=use_refine, phase=phase)
