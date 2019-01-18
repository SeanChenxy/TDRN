# config.py
import os.path
from .voc0712 import VOC_CLASSES, VID_CLASSES, UW_CLASSES

# gets home dir cross platform
home = os.path.expanduser("/")
VOCroot = os.path.join(home,"data/VOCdevkit/")
VIDroot = os.path.join(home,"data/ILSVRC/")
MOT17Detroot = os.path.join(home,"data/MOT/MOT17Det/")
MOT15root = os.path.join(home,"data/MOT/2DMOT2015/")
UWroot = os.path.join(home,"data/UWdevkit/")
COCOroot = os.path.join(home,"data/coco/")

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

dataset_training_cfg = {'VOC0712':([('2007', 'trainval'), ('2012', 'trainval')], len(VOC_CLASSES) + 1, VOCroot),
                        'VIDDET': ('train', len(VID_CLASSES) + 1, VIDroot),
                        'COCO': (['train2014', 'valminusminival2014'], 81, COCOroot)
                        }

#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    # 'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'flip': True,
    'name' : 'VOC_300',
}

VOC_300_RFB = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'flip': True,
    'name' : 'VOC_300_RFB',
}

VOC_320 = {
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_320',
}

VOC_512_RefineDet = {
    'feature_maps': [64, 32, 16, 8],
    'min_dim': 512,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_512_RefineDet',
}

VOC_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],
    'min_dim' : 512,
    'steps' : [8, 16, 32, 64, 128, 256, 512],
    'min_sizes'  : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],
    'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'flip': True,
    'name' : 'VOC_512'
}

MOT_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios': [[1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3],
    #                   [1 / 2, 1 / 3], [1 / 2, 1 / 3]],
    # 'aspect_ratios' : [[2,3,4], [2, 3,4], [2, 3, 4], [2, 3, 4], [2,3,4], [2,3,4]],
    'aspect_ratios': [[1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'flip': False,
    'name' : 'MOT_300',
}

COCO_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [21, 45, 99, 153, 207, 261],
    'max_sizes' : [45, 99, 153, 207, 261, 315],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'flip': True,
    'name': 'COCO_300'
}

COCO_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],
    'min_dim' : 512,
    'steps' : [8, 16, 32, 64, 128, 256, 512],
    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],
    'variance' : [0.1, 0.2],
    'flip': True,
    'clip' : True,
    'name': 'COCO_512'
}

########### Multi-scale 320 ###############
VOC_192 = {
    'feature_maps': [24, 12, 6, 3],
    'min_dim': 192,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_192',
}

VOC_384 = {
    'feature_maps': [48, 24, 12, 6],
    'min_dim': 384,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_384',
}

VOC_448 = {
    'feature_maps': [56, 28, 14, 7],
    'min_dim': 448,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_448',
}

VOC_512_s = {
    'feature_maps': [64, 32, 16, 8],
    'min_dim': 512,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_512_s',
}

VOC_576 = {
    'feature_maps': [72, 36, 18, 9],
    'min_dim': 576,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_576',
}

VOC_704 = {
    'feature_maps': [88, 44, 22, 11],
    'min_dim': 704,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_704',
}

########### Multi-scale 512 ###############
VOC_512_RefineDet_12 = {
    'feature_maps': [80, 40, 20, 10],
    'min_dim': 640,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_512_RefineDet_12',
}

VOC_512_RefineDet_22 = {
    'feature_maps': [152, 76, 38, 19],
    'min_dim': 1216,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_512_RefineDet_22',
}

VOC_512_RefineDet_06 = {
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320,
    'steps': [8, 16, 32, 64],
    'min_sizes': [32, 64, 128, 256],
    'max_sizes': [],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'name': 'VOC_512_RefineDet_06',
}


mb_cfg = {'VOC_300':VOC_300,'VOC_300_RFB':VOC_300_RFB, 'VOC_320':VOC_320, 'VOC_512':VOC_512, 'MOT_300':MOT_300,
          'COCO_300':COCO_300, 'COCO_512':COCO_512, 'VOC_512_RefineDet': VOC_512_RefineDet}
multi_cfg={'192':VOC_192,'320':VOC_320,'384':VOC_384,'448':VOC_448,'512':VOC_512_s,'576':VOC_576, '704':VOC_704}
multi_cfg_512={'320':VOC_512_RefineDet_06,'512':VOC_512_RefineDet,'640':VOC_512_RefineDet_12,'1216':VOC_512_RefineDet_22}