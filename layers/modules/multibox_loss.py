import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, nms

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 device='cuda'):
        super(MultiBoxLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0) # batch
        # priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        # num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        with torch.no_grad():
            loc_t = loc_t.to(self.device)
            conf_t = conf_t.to(self.device)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.sum().float()
        loss_l /=  N
        loss_c /=  N
        return loss_l, loss_c

class seqMultiBoxLoss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, device='cuda',
                 association=False, top_k=200, conf_thresh=0.1, nms_thresh=0.45):
        super(seqMultiBoxLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.association = association
        if self.association:
            self.top_k = top_k
            self.conf_thresh = conf_thresh
            self.nms_thresh = nms_thresh
            self.output = torch.zeros(1, self.num_classes, self.top_k, 5).to(self.device)
            self.past_score = None

    def forward(self, seq_predictions, priors, targets):
        # seq_predictions: [time, batch, (loc, conf, prior)]
        # targets: [batch, time, Var(1,5)]
        seq_loss_l = 0
        seq_loss_c = 0
        loss_association = 0
        self.past_score = None
        for time_step, predictions in enumerate(seq_predictions):
            loc_data, conf_data = predictions
            num = loc_data.size(0) # batch
            # priors = priors[:loc_data.size(1), :]
            num_priors = (priors.size(0))

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.LongTensor(num, num_priors)

            for idx in range(num):
                truths = targets[idx][time_step][:, :-1]
                labels = targets[idx][time_step][:, -1]
                # if priors.dim() == 3:
                #     defaults = priors[idx]
                # else:
                defaults = priors
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)

                loc_t = loc_t.to(self.device)
                conf_t = conf_t.to(self.device)
            # wrap targets
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)
            pos = conf_t > 0

            # Localization Loss (Smooth L1)
            # Shape: [batch,num_priors,4]
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)

            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos+neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

            # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

            N = num_pos.sum().float()
            seq_loss_l += loss_l / N
            seq_loss_c += loss_c / N
            ## consistency
            if self.association:
                conf_data = F.softmax(conf_data.view(-1, self.num_classes), dim=1).view(num, -1, self.num_classes)
                self.output.zero_()
                conf_preds = conf_data.view(num, num_priors,
                                            self.num_classes).transpose(2, 1)
                self.output = self.output.expand(num, self.num_classes, self.top_k, 5).contiguous()

                # Decode predictions into bboxes.
                with torch.no_grad():
                    for i in range(num):
                        decoded_boxes = decode(loc_data[i], defaults, self.variance)
                        # For each class, perform nms
                        conf_scores = conf_preds[i].clone()
                        for cl in range(1, self.num_classes):
                            c_mask = conf_scores[cl].gt(self.conf_thresh)
                            scores = conf_scores[cl][c_mask]
                            if scores.size(0) == 0:
                                continue
                            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                            boxes = decoded_boxes[l_mask].view(-1, 4)
                            # idx of highest scoring and non-overlapping boxes per class
                            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                            self.output[i, cl, :count] = \
                                torch.cat((scores[ids[:count]].unsqueeze(1),
                                           boxes[ids[:count]]), 1)
                    output_score = torch.sum(self.output, dim=2, keepdim=True)[:, :, :, 0]

                    if self.past_score is None:
                        self.past_score = torch.zeros(output_score.size()).to(self.device)
                    else:
                        loss_association += F.smooth_l1_loss(output_score, self.past_score, size_average=False)
                    self.past_score = (self.past_score * time_step + output_score) / (time_step + 1)

        return seq_loss_l/len(seq_predictions), seq_loss_c/len(seq_predictions), loss_association/len(seq_predictions)

