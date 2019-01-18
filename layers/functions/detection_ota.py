import torch
from torch.autograd import Function
import torch.nn.functional as F
from ..box_utils import decode, nms, IoU, cos_similarity, any_same_idx, center_size
import numpy as np

class Detect(Function):
    """At test time, Detect is the final layer of SSD.    Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, tub=0, tub_thresh=1.0, tub_generate_score=0.7, device='cuda'):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]
        self.tub = tub
        self.device = device
        if self.tub > 0:
            self.tubelets = [dict() for _ in range(self.num_classes)]
            self.ides = [None for _ in self.tubelets]
            self.history_max_ides = [-1 for _ in range(self.num_classes)]
            self.tub_thresh = tub_thresh
            self.loss_hold_len = 10
            self.tub_generate_score = tub_generate_score
            self.tub_feature_size = 7
            self.output = torch.zeros(1, self.num_classes, self.top_k, 6, requires_grad=False).to(self.device)
        else:
            self.output = torch.zeros(1, self.num_classes, self.top_k, 5, requires_grad=False).to(self.device)

    def forward(self, loc_data, conf_data, prior_data, feature=None, arm_loc_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            self.output = self.output.expand(num, self.num_classes, self.top_k, 5)
        # Decode predictions into bboxes.
        for i in range(num):
            if arm_loc_data is not None:
                default = decode(arm_loc_data[i], prior_data, self.variance)
                default = center_size(default)
            else:
                default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # a = c_mask.sum()
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    if self.tub > 0:
                        self.delete_tubelets(cl)
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                if self.tub > 0:
                    nms_score = scores[ids[:count]]
                    nms_box = boxes[ids[:count]]
                    nms_feature = list()
                    for obj_box in nms_box:
                        roi_x_min, roi_y_min, roi_x_max, roi_y_max =  int(np.clip(np.floor(obj_box[0] * feature.size(-1)), 0, feature.size(-1))), \
                                                                      int(np.clip(np.floor(obj_box[1] * feature.size(-2)), 0, feature.size(-2))), \
                                                                      int(np.clip(np.ceil(obj_box[2] * feature.size(-1)), 0, feature.size(-1))), \
                                                                      int(np.clip(np.ceil(obj_box[3] * feature.size(-2)), 0, feature.size(-2)))
                        roi_feature = F.upsample(
                            feature[:, :, roi_y_min:roi_y_max, roi_x_min:roi_x_max],
                            (self.tub_feature_size, self.tub_feature_size), mode='bilinear',align_corners=True).view(-1,self.tub_feature_size * self.tub_feature_size * feature.size(1))
                        nms_feature.append(roi_feature)

                    identity = torch.FloatTensor(count).fill_(-1).to(self.device)
                    # matched_score = torch.cuda.FloatTensor(count).fill_(self.tub_thresh)

                    if self.tubelets[cl]:
                        iou = IoU(nms_box, self.tubelets[cl], device=self.device)
                        # iou_max, iou_max_idx = torch.max(iou, dim=1)
                        cos = cos_similarity(nms_feature, self.tubelets[cl], device=self.device)
                        similarity = torch.exp(iou) * cos
                        similarity_max, similarity_max_idx = torch.max(similarity, dim=1)
                        same_idxes_idxes = any_same_idx(similarity_max_idx)
                        for same_idx_idxes in same_idxes_idxes:
                            _, max_idx = torch.max(similarity_max[same_idx_idxes], dim=0)
                            mask = torch.ByteTensor(similarity_max.size()).fill_(0).to(self.device)
                            mask[same_idx_idxes] = 1
                            mask[same_idx_idxes[max_idx]] = 0
                            similarity_max[mask] = 0.

                        # similarity_max_id = self.ides[cl].index_select(0, similarity_max_idx)
                        # tub_conf_list = torch.cuda.FloatTensor([self.tubelets[cl][o][1] for o in similarity_max_id])
                        matched_mask = similarity_max.gt(self.tub_thresh)
                        # matched_score[matched_mask] = similarity_max[matched_mask]
                        if matched_mask.sum():
                            identity[matched_mask] = self.ides[cl].index_select(0, similarity_max_idx[matched_mask]) #similarity_max_id[matched_mask]
                        # identity = self.ides[cl].index_select(0, similarity_max_idx)
                    new_mask = identity.eq(-1)
                    tub_score_mask = nms_score.gt(self.tub_generate_score)
                    generate_mask = new_mask & tub_score_mask
                    if generate_mask.sum() > 0:
                        current = 0 if self.history_max_ides[cl] < 0 else self.history_max_ides[cl] + 1
                        new_id = torch.arange(current, current + generate_mask.sum().float()).to(self.device)
                        self.history_max_ides[cl] = new_id[-1]
                        identity[generate_mask] = new_id
                        #tub_score = torch.zeros(iou_mask.sum()).cuda()
                        # for re in range(iou_mask.sum()):
                        #     t = self.tubelets[cl][identity[iou_mask][re]][0][:,0]
                        #     tub_score[re] = (torch.max(t) + torch.mean(t))/2
                        # nms_score[iou_mask] =  nms_score[iou_mask]*iou_max[iou_mask] + tub_score*(1-iou_max[iou_mask])
                        # nms_score[iou_mask] =  nms_score[iou_mask]*0.5+ tub_score*0.5
                    self.output[i, cl, :count] = \
                          torch.cat((nms_score.unsqueeze(1),
                                   nms_box,
                                   identity.unsqueeze(1)), 1)
                    for det, fea in zip(self.output[i, cl, :count], nms_feature):
                        if det[-1] >=0:
                            tub_info = torch.cat((det[:-1].clone().unsqueeze(0), fea), dim=1)

                            if float(det[-1]) not in self.tubelets[cl]:
                                # self.tubelets[cl][det[-1]] = [tub_info, self.tub_thresh, self.loss_hold_len + 1]
                                self.tubelets[cl][int(det[-1].clone())] = [tub_info, self.loss_hold_len + 1]

                            else:
                                new_tube = torch.cat((tub_info, self.tubelets[cl][int(det[-1])][0]), 0)
                                # new_tub_thresh = np.clip((np.min([self.tubelets[cl][det[-1]][1], ms]) + np.mean([self.tubelets[cl][det[-1]][1], ms]) )/2, 0, 1.5 )
                                # self.tubelets[cl][det[-1]] = [new_tube[:self.tub], new_tub_thresh, self.loss_hold_len + 1] if new_tube.size(
                                #     0) > self.tub else [new_tube, new_tub_thresh, self.loss_hold_len + 1]
                                self.tubelets[cl][int(det[-1].clone())] = [new_tube[:self.tub],self.loss_hold_len + 1] if new_tube.size(
                                    0) > self.tub else [new_tube, self.loss_hold_len + 1]
                    self.delete_tubelets(cl)
                else:
                    self.output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]]), 1)
        flt = self.output.view(-1, self.output.size(-1))
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return self.output

    def init_tubelets(self):
        if self.tub > 0:
            self.tubelets = [dict() for _ in range(self.num_classes)]
            self.ides = [None for _ in self.tubelets]
            self.history_max_ides = [-1 for _ in range(self.num_classes)]

    def delete_tubelets(self, cl):
        delet_list = []
        for ide, tubelet in self.tubelets[cl].items():
            tubelet[-1] -= 1
            if not tubelet[-1]:
                delet_list.append(ide)
        # if not delet_list:
        #     return
        # else:
        for ide in delet_list:
            del self.tubelets[cl][ide]
        self.ides[cl] = torch.FloatTensor(list(self.tubelets[cl].keys())).to(self.device)

