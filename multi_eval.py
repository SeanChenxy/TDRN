"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
from data import VOCroot, VIDroot, UWroot
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, VID_CLASSES, VID_CLASSES_name, UW_CLASSES, mb_cfg, multi_cfg, multi_cfg_512, base_transform
from layers.functions import Detect,PriorBox
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

multi_scale={
        '320':[192, 320, 384, 448, 512, 576, 704],
        '512':[320, 512, 640, 1216]
}

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model_name', default='ssd',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../eval', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--nms_threshold', default=0.45, type=float,
                    help=' nms threshold')
parser.add_argument('--top_k', default=200, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_name', default='VID2017', help='Which dataset')
parser.add_argument('--year', default='2007', help='2007 or 2012')
parser.add_argument('--ssd_dim', default=300, type=int, help='ssd_dim 300 or 512')
parser.add_argument('--backbone', default='VGG16', type=str, help='Backbone')
parser.add_argument('--bn', default=False, type=str2bool, help='Batch norm')
parser.add_argument('--refine', default=False, type=str2bool, help='refine symbol for RefineDet')
parser.add_argument('--deform', default=0, type=int, help='Only work when backbone==RefineDet')
parser.add_argument('--multihead', default=False, type=str2bool, help='Only work when backbone==RefineDet')
parser.add_argument('--rfb', default=0, type=int, help='Only work when backbone==RefineDet')
parser.add_argument('--c7_channel', default=1024, type=int, help='out_channel of Conv7 in VGG')
parser.add_argument('--attention', default=False, type=str2bool, help='attention')
parser.add_argument('--res_attention', default=False, type=str2bool, help='add attention module')
parser.add_argument('--channel_attention', default=True, type=str2bool, help='add attention module')
parser.add_argument('--pm', default=0.0, type=float, help='use predection model or not, the float denotes the channel increment')
parser.add_argument('--set_file_name', default='test', type=str,help='File path to save results')
parser.add_argument('--iteration', default='20000', type=str,help='File path to save results')
parser.add_argument('--model_dir', default='../weights/tssd300_VID2017_b4_s16_SkipShare_preVggExtraLocConf', type=str,help='Path to save model')
parser.add_argument('--detection', default='no', type=str2bool, help='detection or not')
parser.add_argument('--tssd',  default='ssd', type=str, help='ssd or tssd')
parser.add_argument('--gpu_id', default='2,3', type=str,help='gpu id')
parser.add_argument('--tub', default=0, type=int, help='tubelet max size')
parser.add_argument('--tub_thresh', default=0.95, type=float, help='> : generate tubelet')
parser.add_argument('--tub_generate_score', default=0.7, type=float, help='> : generate tubelet')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')
set_type = args.set_file_name.split('_')[0]

dataset_mean = (104, 117, 123)
if args.dataset_name== 'VOC0712':
    annopath = os.path.join(VOCroot, 'VOC2007', 'Annotations', '%s.xml')
    # imgpath = os.path.join(VOCroot, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(VOCroot, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    # YEAR = args.year
    devkit_path = VOCroot + 'VOC' + args.year
    labelmap = VOC_CLASSES
elif args.dataset_name == 'VID2017':
    annopath = os.path.join(VIDroot, 'Annotations', 'VID', set_type, '%s.xml')
    # imgpath = os.path.join(VIDroot, 'Data', 'VID', set_type, '%s.JPEG')
    imgsetpath = os.path.join(VIDroot, 'ImageSets', 'VID', '{:s}.txt')
    devkit_path = VIDroot[:-1]
    labelmap = VID_CLASSES
elif args.dataset_name == 'UW':
    annopath = os.path.join(UWroot, args.year, 'Annotations', set_type, '%s.xml')
    # imgpath = os.path.join(UWroot, 'Data', set_type, '%s.JPEG')
    # imgsetpath = os.path.join(UWroot, 'ImageSets', '{:s}.txt')
    imgsetpath = os.path.join(UWroot, args.year, 'ImageSets', '{:s}.txt')
    devkit_path = UWroot[:-1] + args.year
    labelmap = UW_CLASSES
    dataset_mean = (128, 128, 128)

prior = 'VOC_'+ str(args.ssd_dim)
if 'RefineDet' in args.backbone and args.ssd_dim == 512:
    prior += '_RefineDet'
elif 'RFB' in args.backbone and args.ssd_dim == 300:
    prior += '_RFB'
cfg = mb_cfg[prior]
if args.ssd_dim == 512:
    multi_cfg = multi_cfg_512

ssd_dim = args.ssd_dim
pkl_dir = os.path.join(args.save_folder, args.model_dir.split('/')[-1])
if args.model_dir.split('/')[-1] in ['ssd300_VIDDET', 'ssd300_VIDDET_186', 'ssd300c512_VIDDET','ssd300_VIDDET_512','attssd300_VIDDET_512', 'attssd300_VIDDET_512_atthalf']:
    trained_model = os.path.join(args.model_dir, 'ssd300_VIDDET_' + args.iteration +'.pth')
else:
    if args.tssd in ['tblstm', 'gru']:
        trained_model = os.path.join(args.model_dir, args.model_name+str(args.ssd_dim)+'_' + 'seq'+ args.dataset_name +'_'+ args.iteration +'.pth')
    else:
        trained_model = os.path.join(args.model_dir, args.model_name+str(args.ssd_dim)+'_' + args.dataset_name +'_'+ args.iteration +'.pth')

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        if args.dataset_name == 'VOC0712':
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'comp4_det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True, FPS=None):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    recs = []
    precs = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(args.set_file_name), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        recs += [rec] #if isinstance(rec, float) else rec
        precs += [prec] #if isinstance(prec, float) else prec
        # print(recs)
        # rec_top = rec[args.top_k-1] if len(rec) > args.top_k else rec[-1]
        # prec_top = prec[args.top_k - 1] if len(prec) > args.top_k else prec[-1]
        if args.dataset_name in ['VID2017']:
            print('{} AP = {:.4f}, Rec = {:.4f}, Prec = {:.4f}'.format(VID_CLASSES_name[VID_CLASSES.index(cls)], ap, rec[-1], np.max(prec)))
        else:
            print('{} AP = {:.4f}, Rec = {:.4f}, Prec = {:.4f}'.format(cls, ap, rec[-1], np.max(prec)))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    with open(os.path.join(output_dir, str(np.mean(aps))[2:6])+'.txt', 'w') as res_file:
        res_file.write('CUDA: '+ str(args.cuda)+ '\n')
        res_file.write('model_dir: '+ args.model_dir+ '\n')
        res_file.write('iteration: '+ args.iteration+ '\n')
        res_file.write('model_name: '+ args.model_name+ '\n')
        res_file.write('backbone : '+ args.backbone + '\n')
        if args.backbone in ['RefineDet_VGG']:
            res_file.write('refine : ' + str(args.refine) + '\n')
            res_file.write('deform : ' + str(args.deform) + '\n')
            res_file.write('multi-head : ' + str(args.multihead) + '\n')
        res_file.write('pm: '+ str(args.pm)+ '\n')
        res_file.write('ssd_dim: '+ str(args.ssd_dim)+ '\n')
        res_file.write('tssd: '+ str(args.tssd)+ '\n')
        res_file.write('attention: '+ str(args.attention)+ '\n')
        res_file.write('confidence_threshold: '+ str(args.confidence_threshold)+ '\n')
        res_file.write('nms_threshold: '+ str(args.nms_threshold)+ '\n')
        res_file.write('top_k: '+ str(args.top_k)+ '\n')
        res_file.write('dataset_name: '+ str(args.dataset_name)+ '\n')
        res_file.write('set_file_name: '+ str(args.set_file_name)+ '\n')
        res_file.write('detection: '+ str(args.detection)+ '\n')
        res_file.write('~~~~~~~~~~~~~~~~~\n')
        for i, cls in enumerate(labelmap):
            if args.dataset_name in ['VID2017']:
                res_file.write('{} AP = {:.4f}, Rec = {:.4f}, Prec = {:.4f}\n'.format(VID_CLASSES_name[VID_CLASSES.index(cls)], aps[i], recs[i][-1], np.max(precs[i])))
            else:
                res_file.write('{} AP = {:.4f}, Rec = {:.4f}, Prec = {:.4f}\n'.format(cls, aps[i], recs[i][-1], np.max(precs[i])))
        res_file.write('Mean AP = {:.4f}\n'.format(np.mean(aps)))
        if FPS:
            res_file.write('FPS = {:.4f}\n'.format(FPS))

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_'+args.set_file_name+'.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        if args.dataset_name == 'VID2017':
            lines_tmp = f.readlines()
            lines = []
            for i in range(len(lines_tmp)):
                lines.append(lines_tmp[i].split(' ')[0])
        else:
            lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # from load cachefile load gt, e.g.
        #{'ILSVRC2015_val_00004000/000010': [{'bbox': [241, 114, 451, 233], 'name': 'n02121808'},
        #                                   {'bbox': [106, 59, 384, 245], 'name': 'n02484322'}],
        #'ILSVRC2015_val_00000000/000000': [{'bbox': [416, 6, 605, 171], 'name': 'n01662784'}],
        #'ILSVRC2015_val_00000001/000158': [{'bbox': [644, 137, 903, 384], 'name': 'n01662784'}]}
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0 # the total number of gt objects in a class
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        #e.g. R = [{'name': 'n02121808', 'bbox': [241, 114, 451, 233]}]
        bbox = np.array([x['bbox'] for x in R])
        if args.dataset_name == 'VOC0712':
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        else:
            difficult = np.array([False for _ in R]).astype(np.bool)
        det = [False] * len(R) # False : have not been detected
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    #e.g.: /home/sean/data/ILSVRC/results/det_val_n02834778.txt
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence: greater->smaller
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids) # the total number of det objects in a class
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd): # each det boxes are compared with all
            R = class_recs[image_ids[d]]  # gt
            bb = BB[d, :].astype(float) # det
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float) # gt
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp) # compute top k
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        # original latter all = -1.
        rec = 0.
        prec = 0.
        ap = 0.

    return rec, prec, ap

def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets

def test_net(save_folder, net, dataset, transform, top_k, detector, priors):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:score
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    all_time = 0.
    output_dir = get_output_dir(pkl_dir, args.iteration+'_'+args.dataset_name+'_'+ args.set_file_name)
    det_file = os.path.join(output_dir, 'detections.pkl')
    output_dir = get_output_dir(output_dir, 'multi_test')
    ######################### Multiscale PriorBox #####################
    priorboxes={}
    for v1 in multi_scale[str(ssd_dim)]:
        if not multi_cfg[str(v1)]:
            return ("not included this multi_scale")
        priorbox = PriorBox(multi_cfg[str(v1)])
        img_size = multi_cfg[str(v1)]['min_dim']
        with torch.no_grad():
            priorboxes[str(img_size)] = priorbox.forward().to(device)
    ########################## Detection ##############################
    for i in range(num_images):
        _t['im_detect'].tic()
        image = dataset.pull_image(i)
        h, w, _ = image.shape
        detections_multi={}
        for v in multi_scale[str(ssd_dim)]:
            priors = priorboxes[str(v)]
            ssd_dim_temp = int(v)
            for loop in range(2):
                if (loop == 0):
                    im_trans = base_transform(image, ssd_dim_temp, dataset_mean)
                    im_trans = im_trans[:, :, (2, 1, 0)]
                else:
                    im_f=image.copy()
                    im_f = cv2.flip(im_f, 1)
                    im_trans= base_transform(im_f, ssd_dim_temp, dataset_mean)
                    im_trans = im_trans[:, :, (2, 1, 0)]
                with torch.no_grad():
                    x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                    if 'RefineDet' in args.backbone and args.refine:
                        arm_loc,_, loc, conf = net(x)
                        detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)
                        detections_multi[str(ssd_dim) + '_' + str(v) + '_' + str(loop)] = detections.clone()
                    else:
                        loc, conf = net(x)
                        arm_loc = None
                        detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)
                        detections_multi[str(ssd_dim) + '_' + str(v) + '_' + str(loop)] = detections.clone()
        detect_time = _t['im_detect'].toc(average=False)
        if i > 10:
            all_time += detect_time
    ###################################################################
        for j in range(1, detections.size(1)):
            cls_dets = np.array([])
            for k, d in detections_multi.items():
                dets = d[0, j, :]
                if dets.sum() == 0:
                    continue
                mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
                boxes = dets[:, 1:-1] if dets.size(-1) == 6 else dets[:, 1:]
                if (k[-1] == '1'):
                    boxes[:, 0] = 1 - boxes[:, 0]
                    boxes[:, 2] = 1 - boxes[:, 2]
                    temp_swap = boxes[:, 0].clone()
                    boxes[:, 0] = boxes[:, 2]
                    boxes[:, 2] = temp_swap
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                if k in ['320_192_0', '320_192_1', '512_320_0', '512_320_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.maximum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) > 32)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_320_0', '320_320_1', '512_512_0', '512_512_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.maximum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) > 0)[0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_384_0', '320_384_1', '512_640_0', '512_640_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.minimum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) < 160)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_448_0', '320_448_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.minimum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) < 128)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_512_0', '320_512_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.minimum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) < 96)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_576_0', '320_576_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.minimum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) < 64)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                elif k in ['320_706_0', '320_706_1', '512_1216_0', '512_1216_1']:
                    boxes_np = boxes.cpu().numpy()
                    index_temp = np.where(
                        np.minimum(boxes_np[:, 2] - boxes_np[:, 0] + 1, boxes_np[:, 3] - boxes_np[:, 1] + 1) < 32)[
                        0]
                    if (not index_temp.size):
                        continue
                    else:
                        boxes = boxes[index_temp, :]
                if (index_temp.size == 0):
                    continue
                scores = dets[index_temp, 0].cpu().numpy()
                cls_dets_temp = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                if (cls_dets.size ==0):
                    cls_dets=cls_dets_temp.copy()
                else:
                    cls_dets=np.concatenate((cls_dets,cls_dets_temp),axis=0)
            if (cls_dets.size !=0):
                cls_dets = bbox_vote(cls_dets)
                if( len(cls_dets) !=0):
                    all_boxes[j][i] = cls_dets
        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
    FPS = (num_images - 10) / all_time
    print('FPS:', FPS)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset, FPS=FPS)


def evaluate_detections(box_list, output_dir, dataset, FPS=None):
    write_voc_results_file(box_list, dataset)
    if args.year != '2012':
        do_python_eval(output_dir, FPS=FPS)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # +1 background
    if args.dataset_name == 'VOC0712':
        dataset = VOCDetection(VOCroot, [(args.year, set_type)], BaseTransform(ssd_dim, dataset_mean),
                               AnnotationTransform(dataset_name=args.dataset_name), dataset_name=args.dataset_name )
    elif args.dataset_name == 'VID2017':
        dataset = VOCDetection(VIDroot, set_type, BaseTransform(ssd_dim, dataset_mean),
                               AnnotationTransform(dataset_name=args.dataset_name),
                               dataset_name=args.dataset_name, set_file_name=args.set_file_name)
    elif args.dataset_name == 'UW':
        dataset = VOCDetection(UWroot, [(args.year, set_type)], BaseTransform(ssd_dim, dataset_mean),
                           AnnotationTransform(dataset_name=args.dataset_name),
                           dataset_name=args.dataset_name, set_file_name=args.set_file_name)

    if args.detection:
        if 'RFB' in args.backbone :
            from model.rfbnet_vgg import build_net
            net = build_net('test', ssd_dim, num_classes, bn=args.bn)
        elif 'RefineDet' in args.backbone:
            if args.deform:
                from model.dualrefinedet_vggbn import build_net
                net = build_net('test', size=ssd_dim, num_classes=num_classes,
                                    c7_channel=args.c7_channel, def_groups=args.deform, multihead=args.multihead, bn=args.bn)
            else:
                from model.refinedet_vgg import build_net
                net = build_net('test', size=ssd_dim, num_classes=num_classes, use_refine=args.refine, c7_channel=args.c7_channel)
        else:
            net = None
        print('loading model!', args.model_dir, args.iteration)
        net.load_state_dict(torch.load(trained_model))
        print(net)
        net.eval()
        print('Finished loading model!', args.model_dir, args.iteration,'tub='+str(args.tub), 'tub_thresh='+str(args.tub_thresh), 'tub_score='+str(args.tub_generate_score))
        detector = Detect(num_classes, 0, args.top_k, args.confidence_threshold, args.nms_threshold)
        priorbox = PriorBox(cfg)
        # priorbox=PriorBox(multi_cfg['2.2'])
        with torch.no_grad():
            priors = priorbox.forward().to(device)
        # load data
        net = net.to(device)
        # evaluation
        test_net(args.save_folder, net, dataset,
                     BaseTransform(net.size, dataset_mean), args.top_k, detector, priors)
    else:
        out_dir = get_output_dir(pkl_dir, args.iteration+'_'+args.dataset_name+'_'+ args.set_file_name)
        print('Without detection', out_dir)
        do_python_eval(out_dir)
    print('Finished!', args.model_dir, args.iteration, 'tub='+str(args.tub), 'tub_thresh='+str(args.tub_thresh), 'tub_score='+str(args.tub_generate_score))

