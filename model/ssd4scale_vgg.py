import os
import torch
import torch.nn as nn
from layers import *
from .networks import vgg, vgg_base, ConvOffset2d

class SSD4Scale(nn.Module):

    def __init__(self,size, num_classes=21,  phase='train', c7_channel=1024, bn=True, deform=False):
        super(SSD4Scale, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.phase = phase
        self.bn = bn
        num_box = 3
        self.deform=deform

        # SSD network
        self.backbone = nn.ModuleList(vgg(vgg_base['320'], 3, batch_norm=self.bn, pool5_ds=True, c7_channel=c7_channel))
        self.conv4_3_layer = (23, 33)[self.bn]
        self.conv5_3_layer = (30, 43)[self.bn]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)

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
        if self.deform:
            df_group = 8
            self.offset = nn.ModuleList(
                [nn.Conv2d(num_box * 4,  df_group*2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4,  df_group*2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4,  df_group*2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 nn.Conv2d(num_box * 4,  df_group*2 * 3 * 3, kernel_size=1, stride=1, padding=0),
                 ])

            self.arm_loc = nn.ModuleList(
                [ConvOffset2d(512, num_box * 4, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                 ConvOffset2d(512, num_box * 4, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                 ConvOffset2d(c7_channel, num_box * 4, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                 ConvOffset2d(512, num_box * 4, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                 ])
            self.arm_conf = nn.ModuleList([ConvOffset2d(512, num_box * self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                                       ConvOffset2d(512, num_box * self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                                       ConvOffset2d(c7_channel, num_box * self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                                       ConvOffset2d(512, num_box * self.num_classes, kernel_size=3, stride=1, padding=1, num_deformable_groups=df_group),
                                       ])
        else:
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(c7_channel, num_box*4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, num_box*4, kernel_size=3, stride=1, padding=1),
                                          ])
            self.arm_conf = nn.ModuleList([nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(c7_channel, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(512, num_box*self.num_classes, kernel_size=3, stride=1, padding=1),
                                       ])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, ref_loc=list(), offset_list=list(), ret_loc=False, ret_off=False):
        if self.deform:
            if not offset_list:
                arm_offset_list = []
                for rl, f in zip(ref_loc, self.offset):
                    arm_offset_list.append(f(rl))
            else:
                arm_offset_list = offset_list

        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        if ret_loc:
            arm_loc_for_off = list()
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
        if self.deform:
            for (ams, l, c, f) in zip(arm_sources, self.arm_loc, self.arm_conf, arm_offset_list):
                arm_loc_list.append(l(ams, f).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(c(ams, f).permute(0, 2, 3, 1).contiguous())
        else:
            for (ams, l, c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                loc = l(ams)
                if ret_loc:
                    arm_loc_for_off.append(loc)
                arm_loc_list.append(loc.permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(c(ams).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
        # apply multibox head to source layers

        if self.phase == 'test':
            output = [
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                self.softmax(arm_conf.view(-1, self.num_classes)),  # conf preds
            ]
        else:
            output = [
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                arm_conf.view(arm_conf.size(0), -1, self.num_classes),  # conf preds
            ]
        if ret_loc:
            output.append(arm_loc_for_off)
        if ret_off:
            output.append(arm_offset_list)
        return tuple(output)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_net(phase, size=320, num_classes=21, c7_channel=1024, bn=False, deform=False):
    if size not in [320, 512]:
        print("Error: Sorry only SSD320 and SSD512 is supported currently!")
        return

    return SSD4Scale(size, num_classes=num_classes, phase=phase, c7_channel=c7_channel, bn=bn, deform=deform)
