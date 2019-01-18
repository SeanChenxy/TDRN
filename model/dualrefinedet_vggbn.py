import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from .networks import vgg, vgg_base, ConvOffset2d

class RefineSSD(nn.Module):

    def __init__(self,size, num_classes=21, phase='train', c7_channel=1024, def_groups=1, bn=True, multihead=False, return_feature=False, device='cuda'):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.phase = phase
        self.def_groups=def_groups
        self.bn = bn
        self.multihead = multihead
        self.return_feature = return_feature
        self.device = device
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

        self.arm_loc = nn.ModuleList([nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(c7_channel, num_box*4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                      ])
        self.offset = nn.ModuleList(
                [nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box*4, self.def_groups * 2 * 3 * 3, kernel_size=1, stride=1, padding=0),
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
                [nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4, self.def_groups * 2 * 5 * 5, kernel_size=1, stride=1, padding=0),
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
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        ])
        self.latent_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           ])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        arm_sources = list()
        arm_loc_list = list()
        arm_offset_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()
        if self.multihead:
            arm_offset2_list = list()
        # apply vgg up to conv4_3 relu
        for k in range(self.conv4_3_layer):
            x = self.backbone[k](x)
        s = self.L2Norm_4_3(x)
        arm_sources.append(s)
        ota_feature = None
        if self.return_feature:
            ota_feature = torch.zeros(1,1,s.size(2),s.size(3)).to(self.device)
            for i in range(ota_feature.size(2)):
                for j in range(ota_feature.size(3)):
                    ota_feature[0, 0, i, j] = torch.norm(s[0, :, i, j], 2)

        # apply vgg up to conv5_3
        for k in range(self.conv4_3_layer, self.conv5_3_layer):
            x = self.backbone[k](x)
        s = self.L2Norm_5_3(x)
        arm_sources.append(s)

        # apply vgg up to fc7
        for k in range(self.conv5_3_layer, len(self.backbone)):
            x = self.backbone[k](x)
        arm_sources.append(x)
        # conv6_2
        x = self.extras(x)
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
            for (ob, l, l2, c, c2, f, f2) in zip(obm_sources, self.odm_loc, self.odm_loc_2, self.odm_conf,  self.odm_conf_2, arm_offset_list, arm_offset2_list):
                obm_loc_list.append((l(ob, f)  + l2(ob, f2)).permute(0, 2, 3, 1).contiguous())
                obm_conf_list.append((c(ob, f) + c2(ob, f2)).permute(0, 2, 3, 1).contiguous())
        else:
            for (ob, l, c, f) in zip(obm_sources, self.odm_loc, self.odm_conf, arm_offset_list):
                obm_loc_list.append(l(ob, f).permute(0, 2, 3, 1).contiguous())
                obm_conf_list.append(c(ob, f).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)

        if self.phase == 'test':
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                arm_offset_list, #self.softmax(arm_conf.view(-1, self.num_classes)),  # conf preds
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

def build_net(phase, size=320, num_classes=21, c7_channel=1024, def_groups=1, bn=True, multihead=False, return_feature=False):
    if size not in [320, 512]:
        print("Error: Sorry only SSD320 and SSD512 is supported currently!")
        return

    return RefineSSD(size, num_classes=num_classes, phase=phase, c7_channel=c7_channel, def_groups=def_groups, bn=bn, multihead=multihead, return_feature=return_feature)
