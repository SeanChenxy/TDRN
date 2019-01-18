import matplotlib.pyplot as plt
from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
from data import COCOroot, COCODetection, BaseTransform, mb_cfg, COCO_CLASSES
from layers.functions import Detect,PriorBox
import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import json
import cv2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model_name', default='ssd',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./eval/COCO', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--nms_threshold', default=0.45, type=float,
                    help=' nms threshold')
parser.add_argument('--top_k', default=100, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_name', default='COCO', help='Which dataset')
parser.add_argument('--year', default='2007', help='2007 or 2012')
parser.add_argument('--ssd_dim', default=320, type=int, help='ssd_dim 300 or 512')
parser.add_argument('--backbone', default='RefineDet_VGG', type=str, help='Backbone')
parser.add_argument('--bn', default=True, type=str2bool, help='Batch norm')
parser.add_argument('--refine', default=True, type=str2bool, help='refine symbol for RefineDet')
parser.add_argument('--deform', default=1, type=int, help='Only work when backbone==RefineDet')
parser.add_argument('--multihead', default=True, type=str2bool, help='Only work when backbone==RefineDet')
parser.add_argument('--rfb', default=False, type=str2bool, help='Only work when backbone==RefineDet')
parser.add_argument('--c7_channel', default=1024, type=int, help='out_channel of Conv7 in VGG')
parser.add_argument('--set_file_name', default='test-dev2015', type=str,help='File path to save results')
parser.add_argument('--iteration', default='120', type=str,help='File path to save results')
parser.add_argument('--model_dir', default='./weights040/COCO/ssd320RefineBNMultiDef_COCO', type=str,help='Path to save model')
parser.add_argument('--detection', default='yes', type=str2bool, help='detection or not')
parser.add_argument('--display', default='no', type=str2bool, help='detection or not')
parser.add_argument('--gpu_id', default='2,3', type=str,help='gpu id')

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

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
dataDir = COCOroot[:-1]
dataType = args.set_file_name
output_dir = os.path.join(args.save_folder, args.model_dir.split('/')[-1], args.iteration+'_'+args.set_file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
resFile = '%s/detections_%s_%s_results.json'
resFile = resFile % (output_dir, dataType, args.model_dir.split('/')[-1])

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[1])] = int(ids[0])
    return label_map
label_map = get_label_map(os.path.join(COCOroot, 'coco_labels.txt'))

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


def test_net(save_folder, net, dataset, detector, priors):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    all_time = 0.
    all_forward = 0.
    all_detect = 0.
    det_list=list()
    for i in range(num_images):
        # im, gt, h, w = dataset.pull_item(i)
        im, h, w, img_id = dataset.pull_transformed_image(i)
        if args.display:
            image_draw = dataset.pull_image(i)
        with torch.no_grad():
            x = im.unsqueeze(0).to(device)
            _t['im_detect'].tic()
            if 'RefineDet' in args.backbone and args.refine:
                arm_loc,_, loc, conf = net(x)
            else:
                loc, conf = net(x)
                arm_loc = None
            forward_time = _t['im_detect'].toc(average=False)
            _t['im_detect'].tic()
            detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)
            detect_time = _t['im_detect'].toc(average=False)
        if i>10:
            all_time += detect_time + forward_time
            all_forward += forward_time
            all_detect += detect_time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if dets.sum() == 0:
                continue
            mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
            boxes = dets[:, 1:-1] if dets.size(-1)==6 else dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            # boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2] # to x,y,w,h
            boxes_np = boxes.cpu().numpy()

            for b, s in zip(boxes_np, scores):
                # b = [float('{:.2f},{:.2f},{:.2f},{:.2f}'.format(bl)) for bl in b]
                det_list.append({'image_id': img_id, 'category_id': label_map[j],
                                 'bbox': [float('{:.1f}'.format(b[0])), float('{:.1f}'.format(b[1])),float('{:.1f}'.format(b[2]-b[0]+1)),float('{:.1f}'.format(b[3]-b[1]+1))],
                                 'score': float('{:.2f}'.format(s))})
                if s > 0.5:
                    # out.append([x_min, y_min, x_max, y_max, j - 1, score.cpu().numpy(), identity])
                    # if save_dir:
                    #     results_file.write(line[:-1] + ' ' + str(j) + ' ' + str(score_np) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + '\n')
                    if args.display:
                        cv2.rectangle(image_draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), thickness=1)
                        put_str = str(COCO_CLASSES[j-1] + ':' + str(np.around(s, decimals=2)))
                        cv2.putText(image_draw, put_str,
                            (int(b[0]) + 10, int(b[1]) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color=(0, 0, 255), thickness=1)
        if args.display:
            cv2.imshow('frame', image_draw)
            cv2.waitKey(0)
        if i % (int(num_images / 100)) == 0:
            print('im_detect: {:d}/{:d} {:.5f}s, {:.5f}s, {:.5f}s'.format(i + 1,
                                                                  num_images, forward_time, detect_time,
                                                                  forward_time + detect_time))

    with open(resFile, 'w') as f:
        # det_dict_json= json.dumps
        # det_dict_json = [{'1':1,'2':2}]
        json.dump(det_list, f)
    FPS = (num_images - 10) / all_time
    FPS_forward = (num_images - 10) / all_forward
    FPS_detect = (num_images - 10) / all_detect
    print('forward: ', FPS_forward, 'detect: ', FPS_detect, 'FPS:', FPS)

    if args.set_file_name.find('test') == -1:
        print('Evaluating detections')
        evaluate_detections(FPS=(FPS_forward, FPS_detect, FPS))

def evaluate_detections(FPS=None):

    print('Running demo for *%s* results.' % (annType))
    # initialize COCO ground truth api
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    print(annFile)
    cocoGt = COCO(annFile)
    # initialize COCO detections api
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = cocoGt.getImgIds()
    # imgIds = imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    means = cocoEval.summarize()
    with open(os.path.join(output_dir, str(int(means[0]*10000))+'.txt'), 'w') as res_file:
        res_file.write('CUDA: '+ str(args.cuda)+ '\n')
        res_file.write('model_dir: '+ args.model_dir+ '\n')
        res_file.write('iteration: '+ args.iteration+ '\n')
        res_file.write('model_name: '+ args.model_name+ '\n')
        res_file.write('backbone : '+ args.backbone + '\n')
        if args.backbone in ['RefineDet_VGG']:
            res_file.write('refine : ' + str(args.refine) + '\n')
            res_file.write('deform : ' + str(args.deform) + '\n')
            res_file.write('multi-head : ' + str(args.multihead) + '\n')
        res_file.write('ssd_dim: '+ str(args.ssd_dim)+ '\n')
        res_file.write('confidence_threshold: '+ str(args.confidence_threshold)+ '\n')
        res_file.write('nms_threshold: '+ str(args.nms_threshold)+ '\n')
        res_file.write('top_k: '+ str(args.top_k)+ '\n')
        res_file.write('dataset_name: '+ str(args.dataset_name)+ '\n')
        res_file.write('set_file_name: '+ str(args.set_file_name)+ '\n')
        res_file.write('detection: '+ str(args.detection)+ '\n')
        res_file.write('~~~~~~~~~~~~~~~~~\n')
        res_file.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.4f}\n'.format(means[0]))
        res_file.write('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.4f}\n'.format(means[1]))
        res_file.write('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.4f}\n'.format(means[2]))
        res_file.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.4f}\n'.format(means[3]))
        res_file.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.4f}\n'.format(means[4]))
        res_file.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.4f}\n'.format(means[5]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.4f}\n'.format(means[6]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.4f}\n'.format(means[7]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.4f}\n'.format(means[8]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.4f}\n'.format(means[8]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.4f}\n'.format(means[10]))
        res_file.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.4f}\n'.format(means[11]))
        if FPS:
            for i, f in enumerate(FPS):
                res_file.write(str(i)+': FPS = {:.4f}\n'.format(f))

if __name__ == '__main__':
    if args.detection:
        num_classes = 81  # +1 background
        prior = 'VOC_' + str(args.ssd_dim)
        if 'RefineDet' in args.backbone and args.ssd_dim == 512:
            prior += '_RefineDet'
        elif 'RFB' in args.backbone and args.ssd_dim == 300:
            prior += '_RFB'
        cfg = mb_cfg[prior]
        dataset_mean = (104, 117, 123)
        ssd_dim = args.ssd_dim
        dataset = COCODetection(COCOroot, year=args.year, image_sets=[args.set_file_name,], transform=BaseTransform(ssd_dim, dataset_mean), phase='test')

        if 'MobNet' in args.backbone:
            if args.deform:
                from model.dualrefinedet_mobilenet import build_net
                net = build_net('test', size=ssd_dim, num_classes=num_classes,
                                    def_groups=args.deform, multihead=args.multihead)
            else:
                from model.refinedet_mobilenet import build_net
                net = build_net('test', size=ssd_dim, num_classes=num_classes, use_refine=args.refine)
        elif args.deform:
            from model.dualrefinedet_vggbn import build_net
            net = build_net('test', size=ssd_dim, num_classes=num_classes,
                            c7_channel=args.c7_channel, def_groups=args.deform, multihead=args.multihead, bn=args.bn)
        else:
            from model.refinedet_vgg import build_net
            net = build_net('test', size=ssd_dim, num_classes=num_classes, use_refine=args.refine,
                            c7_channel=args.c7_channel, bn=args.bn, multihead=args.multihead)

        print('loading model!', args.model_dir, args.iteration)
        if '.pth' in args.model_dir.split('/')[-1]:
            pkl_dir = os.path.join(args.save_folder, args.model_dir.split('/')[-2])
            trained_model = args.model_dir
        else:
            trained_model = os.path.join(args.model_dir, args.model_name+str(args.ssd_dim)+'_COCO_' + args.iteration +'.pth')
        net.load_state_dict(torch.load(trained_model))
        net.eval()
        print('Finished loading model!', args.model_dir, args.iteration)
        detector = Detect(num_classes, 0, args.top_k, args.confidence_threshold, args.nms_threshold)
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward().to(device)
        # load data
        net = net.to(device)
        # evaluation
        test_net(args.save_folder, net, dataset, detector, priors)
    else:
        evaluate_detections()
        # evaluate_coco()
    print('Finished!', args.model_dir, args.iteration)
