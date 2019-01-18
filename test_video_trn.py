import torch
import torch.backends.cudnn as cudnn
from data import base_transform, mb_cfg
from layers.functions import Detect,PriorBox
from layers.box_utils import decode, center_size
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
cuda = True
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True

#################### Parameter Setting ##################
dataset_name = 'VID2017'
static_dir = '../weights/model1.pth'
trn_dir = '../weights/model2.pth'
backbone = 'VGG'
ssd_dim=320
bn = True
c7_channel = 1024
interval=4
deform=True
# save_dir = './demo/TRN'
save_dir = None
##########################################################
if dataset_name == 'VID2017':
    from data import VID_CLASSES, VID_CLASSES_name
    video_name='/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00047000.mp4'
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
    confidence_threshold = 0.5
    nms_threshold = 0.3
    top_k = 200
else:
    raise ValueError("dataset [%s] not recognized." % dataset_name)
if save_dir:
    save_dir = save_dir+'/'+video_name.split('/')[-1].split('.')[0]
if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
prior = 'VOC_' + str(ssd_dim)
cfg = mb_cfg[prior]

def main():
    mean = (104, 117, 123)
    if 'FPN' in backbone:
        from model.refinedet_vgg import build_net
        static_net = build_net('test', size=ssd_dim, num_classes=num_classes, c7_channel=c7_channel, bn=bn)
        net = build_net('test', size=ssd_dim, num_classes=num_classes, c7_channel=c7_channel, bn=bn)
    else:
        from model.ssd4scale_vgg import build_net
        static_net = build_net('test', size=ssd_dim, num_classes=num_classes, c7_channel=c7_channel, bn=bn)
        net = build_net('test', size=ssd_dim, num_classes=num_classes, c7_channel=c7_channel, bn=bn, deform=deform)

    print('loading model!')
    static_net.load_state_dict(torch.load(static_dir))
    static_net.eval()
    static_net = static_net.to(device)
    net.load_state_dict(torch.load(trn_dir))
    net.eval()
    net = net.to(device)
    print('Finished loading model!', static_dir, trn_dir)
    detector = Detect(num_classes, 0, top_k, confidence_threshold, nms_threshold)
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward().to(device)

    frame_num = 0
    cap = cv2.VideoCapture(video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    size = (640, 480)
    if save_dir:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        record = cv2.VideoWriter(os.path.join(save_dir,video_name.split('/')[-1].split('.')[0]+'.avi'), fourcc, cap.get(cv2.CAP_PROP_FPS), size)
    # static_flag = True
    offset_list = list()
    ref_loc = list()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame_draw = frame.copy()
        im_trans = base_transform(frame, ssd_dim, mean)
        with torch.no_grad():
            x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            if frame_num%interval == 0:
            # if static_flag:
                static_out = static_net(x, ret_loc=deform)
                priors_static = center_size(decode(static_out[0][0], priors, [0.1,0.2]))
                if deform:
                    ref_loc = static_out[2]  # [o * args.loose for o in static_out[2]]
                    offset_list = list()
            out = net(x, ref_loc=ref_loc, offset_list=offset_list, ret_off=(False, True)[deform and not offset_list])
            detections = detector.forward(out[0], out[1], priors_static, scale=torch.cuda.FloatTensor([w,h,w,h]))
            if len(detections) == 3:
                offset_list = out[2]
                ref_loc = list()
            # if static_flag:
            #     ref_mask = mask.clone()mask
            #     print('static')
            #     static_flag = False
            # else:
            #     time1 = time.time()
            #     s_score = (mask * ref_mask).sum().float() / (mask + ref_mask).sum().float()
            #     static_flag = (False, True)[s_score<0.45]
            #     time2 = time.time()
            #     print(s_score, 'match time:', time2-time1)
        out = list()
        for j in range(1, detections.size(1)):
            if detections[0, j, :, :].sum() == 0:
                continue
            for k in range(detections.size(2)):
                dets = detections[0, j, k, :]
                if dets.sum() == 0:
                    continue
                boxes = dets[1:-1] if dets.size(0) == 6 else dets[1:]
                identity = dets[-1] if dets.size(0) == 6 else -1
                x_min = int(boxes[0] * w)
                x_max = int(boxes[2] * w)
                y_min = int(boxes[1] * h)
                y_max = int(boxes[3] * h)

                score = dets[0]
                if score > confidence_threshold:
                    put_str =  VID_CLASSES_name[j-1] +':'+ str(np.around(score, decimals=2)).split('(')[-1].split(',')[0][:4]
                    color = (255,0,0)
                    cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
                    cv2.putText(frame_draw, put_str,
                            (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color=color, thickness=1)
        print(str(frame_num))
        frame_num += 1
        frame_show = cv2.resize(frame_draw, size)
        cv2.imshow('frame', frame_show)# 255* mask.cpu().numpy())
        if save_dir:
            record.write(frame_show)
        ch = cv2.waitKey(1)
        if ch == 32:
        # if frame_num % 1 ==0:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if save_dir:
                        print('save: ', frame_num)
                        torch.save(out, os.path.join(save_dir, '_%s.pkl' % str(frame_num)))
                        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(frame_num)), frame)
                elif in_ch == 32:
                    break

    cap.release()
    if save_dir:
        record.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

