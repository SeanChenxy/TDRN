import torch
import torch.backends.cudnn as cudnn
from data import base_transform, mb_cfg
from layers.functions import Detect,PriorBox
import os
import numpy as np
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
cuda = True
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True

#################### Parameter Setting ##################
dataset_name = 'UW'
model_dir = './weights/model.pth'
backbone = 'RefineDet_VGG'
ssd_dim=320
bn = True
refine = True
deform = 1
multihead = True
attention = False
channel_attention = False
res_attention = False
c7_channel = 1024
tssd = 'ssd'
pm = 0
tub = 0
tub_thresh = 1
tub_generate_score = 0.1
# save_dir = './demo'
save_dir = None
##########################################################
if save_dir and not os.path.exists(save_dir):
    os.mkdir(save_dir)

if dataset_name == 'VID2017':
    from data import VID_CLASSES, VID_CLASSES_name
    video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00027000.mp4'
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
    prior = 'v2'
    confidence_threshold = 0.5
    nms_threshold = 0.5
    top_k = 200
else:
    raise ValueError("dataset [%s] not recognized." % dataset_name)

prior = 'VOC_'+ str(ssd_dim)
if 'RefineDet' in backbone and ssd_dim == 512:
    prior += '_RefineDet'
elif 'RFB' in backbone and ssd_dim == 300:
    prior += '_RFB'
cfg = mb_cfg[prior]

def main():
    mean = (104, 117, 123)
    trained_model = model_dir
    if 'RFB' in backbone:
        from model.rfbnet_vgg import build_net
        net = build_net('test', ssd_dim, num_classes, bn=bn)
    elif 'RefineDet' in backbone:
        if deform:
            from model.dualrefinedet_vggbn import build_net
            net = build_net('test', size=ssd_dim, num_classes=num_classes, return_feature=(False,True)[tub>0],
                            c7_channel=c7_channel, def_groups=deform, multihead=multihead, bn=bn)
        else:
            from model.refinedet_vgg import build_net
            net = build_net('test', size=ssd_dim, num_classes=num_classes, use_refine=refine,
                            c7_channel=c7_channel, bn=bn, multihead=multihead)
    else:
        net = None
    print('loading model!')
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    net = net.to(device)
    print('Finished loading model!', model_dir)
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

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        frame_num += 1
        im_trans = base_transform(frame, ssd_dim, mean)
        with torch.no_grad():
            x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            if tssd == 'ssd':
                if 'RefineDet' in backbone and refine:
                    time3 = time.time()
                    arm_loc, ota_feature, loc, conf = net(x)
                    time4 = time.time()
                    print(time4-time3)
                else:
                    loc, conf = net(x)
                    arm_loc, ota_feature = None, None
                detections = detector.forward(loc, conf, priors, arm_loc_data=arm_loc, feature=ota_feature)
            else:
                detections, state, att_map = net(x, state)
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
                    if dataset_name in ['MOT15']:
                        put_str = str(int(identity))
                        if identity in [34]:
                            color = (0, 0, 255)
                        elif identity in [35]:
                            color = (0, 200, 0)
                        elif identity in [58]:
                            color = (255, 0, 255)
                        else:
                            color = (255, 0, 0)
                    elif dataset_name in ['VID2017']:
                        put_str = str(int(identity))+':'+VID_CLASSES_name[j-1] +':'+ str(np.around(score, decimals=2))
                    elif dataset_name in ['UW']:
                        put_str = str(np.around(score.cpu().numpy(), decimals=2))
                        if j == 1:
                            color = (255,255,0)
                        elif j == 2:
                            color = (0,0,255)
                        elif j == 3:
                            color = (0,255,255)
                        else:
                            color = (0, 128, 255)

                cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
                # cv2.fillConvexPoly(frame_draw, np.array(
                #     [[x_min - 1, y_min], [x_min - 1, y_min - 50], [x_max + 1, y_min - 50], [x_max + 1, y_min]], np.int32), color)
                cv2.putText(frame_draw, put_str,
                            (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color=color, thickness=1)
        print(str(frame_num))
        frame_show = cv2.resize(frame_draw, size)
        cv2.imshow('frame', frame_show)
        if save_dir:
            record.write(frame_show)
        ch = cv2.waitKey(1)
        if ch == 32:
        # if frame_num in [44]:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if save_dir:
                        print('save: ', frame_num)
                        torch.save(out, os.path.join(save_dir, tssd+'_%s.pkl' % str(frame_num)))
                        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(frame_num)), frame)
                elif in_ch == 32:
                    break

    cap.release()
    if save_dir:
        record.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

