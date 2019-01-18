import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from .networks import ConvOffset2d, conv_dw

class RefineSSD(nn.Module):

    def __init__(self,size ,num_classes=21, phase='train', def_groups=1, multihead=False):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.phase = phase
        self.def_groups=def_groups
        self.multihead=multihead
        num_box = 3
        # SSD network
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
            conv_dw(1024, 1024, 1), # p3
        ])
        self.L2Norm_4_3 = L2Norm(512, 20)
        self.L2Norm_5_3 = L2Norm(1024, 8)

        self.extras = nn.ModuleList([nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        conv_dw(256, 512, 2)),
                                      nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True),
                                          conv_dw(256, 512, 2))
                                     ])
        ################################################################################################
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))

        self.arm_loc = nn.ModuleList([nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Conv2d(1024, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1, bias=False),
                                      ])
        self.offset = nn.ModuleList(
                [nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0, bias=False),
                 ])

        self.odm_loc = nn.ModuleList([ConvOffset2d(256, num_box*4, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                      ConvOffset2d(256, num_box*4, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                      ConvOffset2d(256, num_box*4, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                      ConvOffset2d(256, num_box*4, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                      ])
        self.odm_conf = nn.ModuleList([ConvOffset2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                       ConvOffset2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                       ConvOffset2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                       ConvOffset2d(256, num_box*self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=self.def_groups),
                                       ])
        if self.multihead:
            # 5x5
            self.offset2 = nn.ModuleList(
                [nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0, bias=False),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0, bias=False),
                 ])
            self.odm_loc_2 = nn.ModuleList([ConvOffset2d(256, num_box * 4, kernel_size=5, stride=1, padding=2,
                                                         dilation=1, num_deformable_groups=self.def_groups),
                                            ConvOffset2d(256, num_box * 4, kernel_size=5, stride=1, padding=2,
                                                         dilation=1, num_deformable_groups=self.def_groups),
                                            ConvOffset2d(256, num_box * 4, kernel_size=5, stride=1, padding=2,
                                                         dilation=1, num_deformable_groups=self.def_groups),
                                            ConvOffset2d(256, num_box * 4, kernel_size=5, stride=1, padding=2,
                                                         dilation=1, num_deformable_groups=self.def_groups),
                                            ])
            self.odm_conf_2 = nn.ModuleList(
                [ConvOffset2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2, dilation=1,
                              num_deformable_groups=self.def_groups),
                 ConvOffset2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2, dilation=1,
                              num_deformable_groups=self.def_groups),
                 ConvOffset2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2, dilation=1,
                              num_deformable_groups=self.def_groups),
                 ConvOffset2d(256, num_box * self.num_classes, kernel_size=5, stride=1, padding=2, dilation=1,
                              num_deformable_groups=self.def_groups),
                 ])

        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)),
                                           nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)),
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)),
                                           ])
        self.up_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
                                        ])
        self.latent_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           ])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        arm_sources = list()
        arm_loc_list = list()
        arm_offset_list = list()
        odm_loc_list = list()
        odm_conf_list = list()
        odm_sources = list()
        if self.multihead:
            arm_offset2_list = list()

        for i in range(12):
            x = self.backbone[i](x)
        arm_sources.append(self.L2Norm_4_3(x))
        for i in range(12, len(self.backbone)):
            x = self.backbone[i](x)
        arm_sources.append(self.L2Norm_5_3(x))

        for v in self.extras:
            x = v(x)
            arm_sources.append(x)

        if self.multihead:
            for (a, l, f, f2) in zip(arm_sources, self.arm_loc,self.offset, self.offset2):
                loc_a = l(a)
                arm_loc_list.append(loc_a.permute(0, 2, 3, 1).contiguous())
                arm_offset_list.append(f(loc_a))
                arm_offset2_list.append(f2(loc_a))
        else:
            for (a, l, f) in zip(arm_sources, self.arm_loc, self.offset):
                loc_a = l(a)
                arm_loc_list.append(loc_a.permute(0, 2, 3, 1).contiguous())
                arm_offset_list.append(f(loc_a))
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
        x = self.last_layer_trans(x)
        odm_sources.append(x)

        # get transformed layers
        trans_layer_list = list()
        for (x_t, t) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t(x_t))
        # fpn module
        trans_layer_list.reverse()
        for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layers):
            x = F.relu(l(F.relu(u(x) + t, inplace=True)), inplace=True)
            odm_sources.append(x)
        odm_sources.reverse()
        if self.multihead:
            for (od, l, l2, c, c2, f, f2) in zip(odm_sources, self.odm_loc, self.odm_loc_2, self.odm_conf,  self.odm_conf_2, arm_offset_list, arm_offset2_list):
                odm_loc_list.append((l(od, f)  + l2(od, f2)).permute(0, 2, 3, 1).contiguous())
                odm_conf_list.append((c(od, f) + c2(od, f2)).permute(0, 2, 3, 1).contiguous())
        else:
            for (ob, l, c, f) in zip(odm_sources, self.odm_loc, self.odm_conf, arm_offset_list):
                odm_loc_list.append(l(ob, f).permute(0, 2, 3, 1).contiguous())
                odm_conf_list.append(c(ob, f).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf_list], 1)

        # apply multibox head to source layers
        if self.phase == 'test':
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                None, #self.softmax(arm_conf.view(-1, self.num_classes)),  # conf preds
                obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                None, # arm_conf.view(arm_conf.size(0), -1, self.num_classes),  # conf preds
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

def build_net(phase, size=320, num_classes=21, def_groups=1, multihead=False):
    if size not in [320, 512]:
        print("Error: Sorry only SSD320 and SSD512 is supported currently!")
        return
    return RefineSSD(size, num_classes=num_classes, phase=phase, def_groups=def_groups, multihead=multihead)
