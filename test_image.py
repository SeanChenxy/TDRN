import torch
import torch.backends.cudnn as cudnn
from data import base_transform, VOC_CLASSES, VOCroot, UW_CLASSES, UWroot, mb_cfg, COCOroot
from layers.functions import Detect,PriorBox
import os
import numpy as np
import cv2
import json

#################### Parameter Setting ########################
backbone = 'RefineDet_VGG'
ssd_dim=512
bn = True
refine = True
deform = 1
multihead = True
attention = False
model_dir = './weights040/COCO/ssd512RefineBNMultiDef5_COCO'
iteration = 120
confidence_threshold = 0.5
nms_threshold = 0.45
top_k = 200
dataset = 'COCO'
image_list = 'test-dev2015'
year = 'test2015'
save_folder = './demo/DRN/' + dataset
save_dir = os.path.join(save_folder, model_dir.split('/')[-1], str(iteration)+ '_'+ image_list)
# save_dir = None
resfile_name = 'detections_' + image_list +'_'+ model_dir.split('/')[-1].split('_')[0]+str(iteration) + '_results'
trained_model = os.path.join(model_dir, 'ssd'+str(ssd_dim)+'_'+dataset+'_'+str(iteration)+'.pth')
display = True
gpu_id = '6'
cuda = True
##############################################################
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True

if 'VOC' in dataset:
    labelmap = VOC_CLASSES
    num_classes = len(VOC_CLASSES)+1
    root = VOCroot
    img_set = os.path.join(root, 'VOC'+year, 'ImageSets', 'Main', image_list+'.txt')
    img_root = os.path.join(root, 'VOC'+year, 'JPEGImages')
elif 'UW' in dataset:
    labelmap = UW_CLASSES
    num_classes = len(UW_CLASSES)+1
    root = UWroot
    img_set = os.path.join(root, year, 'ImageSets', image_list+'.txt')
    img_root = os.path.join(root, year, 'Data', image_list.split('_')[0])
elif 'COCO' in dataset:
    labelmap = {}
    class_name={}
    labels = open(os.path.join(COCOroot, 'coco_labels.txt'), 'r')
    for line in labels:
        ids = line.split(',')
        labelmap[int(ids[1])] = int(ids[0])
        class_name[int(ids[1])] = ids[2][:-1]
    num_classes = 81
    root = COCOroot
    img_set = os.path.join(root, 'ImageSets', image_list+'.txt')
    img_root = os.path.join(root, 'images', year)
    det_list = list()

prior = 'VOC_'+ str(ssd_dim)
if 'RefineDet' in backbone and ssd_dim == 512:
    prior += '_RefineDet'
elif 'RFB' in backbone and ssd_dim == 300:
    prior += '_RFB'
cfg = mb_cfg[prior]

if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = '.json' if dataset=='COCO' else '.txt'
    results_file = open(os.path.join(save_dir, resfile_name + suffix), 'w')

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    mean = (104, 117, 123)
    print('loading model!')
    if deform:
        from model.dualrefinedet_vggbn import build_net
        net = build_net('test', size=ssd_dim, num_classes=num_classes,
                        c7_channel=1024, def_groups=deform, multihead=multihead, bn=bn)
    else:
        from model.refinedet_vgg import build_net
        net = build_net('test', size=ssd_dim, num_classes=num_classes, use_refine=refine,
                        c7_channel=1024, bn=bn)
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    print('Finished loading model!', trained_model)
    net = net.to(device)
    detector = Detect(num_classes, 0, top_k, confidence_threshold, nms_threshold)
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward().to(device)
    for i, line in enumerate(open(img_set, 'r')):
        # if i==10:
        #     break
        if 'COCO' in dataset:
            image_name = line[:-1]
            image_id = int(image_name.split('_')[-1])
        elif 'VOC' in dataset:
            image_name = line[:-1]
            image_id = -1
        else:
            image_name, image_id = line.split(' ')
            image_id = image_id[:-1]
        print(i, image_name, image_id)
        image_path = os.path.join(img_root, image_name +'.jpg')
        image = cv2.imread(image_path, 1)
        h,w,_ = image.shape
        image_draw = cv2.resize(image.copy(), (640,480))
        im_trans = base_transform(image, ssd_dim, mean)
        ######################## Detection ########################
        with torch.no_grad():
            x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            if 'RefineDet' in backbone and refine:
                arm_loc,_, loc, conf = net(x)
            else:
                loc, conf = net(x)
                arm_loc = None
            detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc)
        ############################################################
        out = list()
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if dets.sum() == 0:
                continue
            mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
            boxes = dets[:, 1:-1] if dets.size(-1) == 6 else dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            for b, s in zip(boxes_np, scores):
                if save_dir:
                    out.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), j - 1, s])
                    if 'COCO' in dataset:
                        det_list.append({'image_id': image_id, 'category_id': labelmap[j],
                             'bbox': [float('{:.1f}'.format(b[0])), float('{:.1f}'.format(b[1])),
                                      float('{:.1f}'.format(b[2] - b[0] + 1)),
                                      float('{:.1f}'.format(b[3] - b[1] + 1))],
                             'score': float('{:.2f}'.format(s))})
                    else:
                        results_file.write(str(image_id) + ' ' + str(j) + ' ' + str(s) + ' ' + str(np.around(b[0],2)) + ' ' + str(np.around(b[1],2)) + ' ' + str(np.around(b[2],2)) + ' ' + str(np.around(b[3],2)) + '\n')
                if display:
                    cv2.rectangle(image_draw, (int(b[0]/w*640), int(b[1]/h*480)), (int(b[2]/w*640), int(b[3]/h*480)), (0,255,0), thickness=1)

                    cls = class_name[j] if 'COCO' in dataset else str(labelmap[j-1])
                    put_str = cls + ':' + str(np.around(s, decimals=2))
                    cv2.putText(image_draw, put_str,
                        (int(b[0]/w*640), int(b[1]/h*480)-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color=(0,255,0), thickness=1)
        if display:
            cv2.imshow('frame', image_draw)
            ch = cv2.waitKey(0)
            if ch == 115:
                if save_dir:
                    print('save: ', line)
                    torch.save(out, os.path.join(save_dir, '%s.pkl' % str(line[:-1])))
                    cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(line[:-1])), image)
                    cv2.imwrite(os.path.join(save_dir, '%s_box.jpg' % str(line[:-1])), image_draw)

    cv2.destroyAllWindows()
    if save_dir:
        if dataset == 'COCO':
            json.dump(det_list, results_file)
        results_file.close()


if __name__ == '__main__':
    main()

