from __future__ import print_function

import os
import torch
import numpy as np

import torch.nn.functional as F

def save_checkpoint(state, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)

class SegMetric(object):
    def __init__(self, use_ignore=True, ignore_label=255):
        self._nclass = 0
        self._names = ['pixel-acc', 'mean-recall', 'mean-acc', 
                       'mean-iou', 'rcls', 'accs', 'ious']
        self._use_ignore = use_ignore
        self._ignore_label = ignore_label

    def set_nclass(self, number_of_classes):
        self._nclass = number_of_classes
        self._tp = np.zeros(self._nclass)
        self._fp = np.zeros(self._nclass)
        self._fn = np.zeros(self._nclass)
        self._num_inst = np.zeros(self._nclass)
        self._cm = np.zeros((self._nclass, self._nclass))

    def reset(self):
        self._tp = np.zeros(self._nclass)
        self._fp = np.zeros(self._nclass)
        self._fn = np.zeros(self._nclass)
        self._num_inst = np.zeros(self._nclass)
        self._cm = np.zeros((self._nclass, self._nclass))

    def update(self, labels, pred_logits):
        preds=F.softmax(pred_logits, dim=1)
        seg_preds = preds.argmax(dim=1) 
        for pred, label in zip(seg_preds, labels):
            predict = pred.detach().cpu().numpy().astype('int64')
            target = label.detach().cpu().numpy().astype('int64')
            if self._use_ignore:
                mask = target != self._ignore_label
                target = target[mask]
                predict = predict[mask]
            for i in range(self._nclass):
                self._tp[i] += ((target == i) & (predict == i)).sum()
                self._fp[i] += ((target != i) & (predict == i)).sum()
                self._fn[i] += ((target == i) & (predict != i)).sum()
                self._num_inst[i] += (target == i).sum()
                for j in range(self._nclass):
                    self._cm[i][j] += ((target == i) & (predict == j)).sum()

    def get(self):
        pixel_acc = self._tp.sum() / self._num_inst.sum()
        rcls = np.divide(self._tp, self._num_inst,
                         out=np.full_like(self._tp, 0.0), where=self._tp != 0)
        accs = np.divide(self._tp, self._tp + self._fp,
                         out=np.full_like(self._tp, 0.0), where=self._tp != 0)
        ious = np.divide(self._tp, self._tp + self._fp + self._fn,
                         out=np.full_like(self._tp, 0.0), where=self._tp != 0)

        values = [pixel_acc,
                  rcls[np.logical_not(np.isnan(rcls))].mean(),
                  accs[np.logical_not(np.isnan(accs))].mean(),
                  ious[np.logical_not(np.isnan(ious))].mean(),
                  rcls, accs, ious]
        return (self._names, values)

    def get_matrix(self):
        # matrix = np.divide(self._cm, self._num_inst.T[None, :].repeat(self._nclass, axis=0),
        #           out=np.full_like(self._cm, 0.0), where=self._num_inst != 0)
        matrix = np.divide(self._cm, self._num_inst[:, None],
                           out=np.full_like(self._cm, 0.0), where=self._num_inst[:, None] != 0)

        # print "matrix: \n{}".format(matrix)
        # print "matrix sum: {}".format(np.sum(matrix, axis=0))
        # print "matrix sum: {}".format(np.sum(matrix, axis=1))
        # print np.sum(self._cm,axis=0) == self._num_inst
        # print np.sum(self._cm, axis=1) == self._num_inst
        return matrix

    def get_miou(self):
        ious = np.divide(self._tp, self._tp + self._fp + self._fn,
                         out=np.full_like(self._tp, 0.0), where=self._tp != 0)
        mean_iou = ious[np.logical_not(np.isnan(ious))].mean()
        return ious.tolist(), mean_iou


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.avg = val

def evaluate_seg(pred, gt, fg_thr=0.5):
    pred_binary = (pred >= fg_thr).float().cuda()
    pred_binary_inverse = (pred_binary == 0).float().cuda()

    gt_binary = (gt >= fg_thr).float().cuda()
    gt_binary_inverse = (gt_binary == 0).float().cuda()

    MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
    TP = pred_binary.mul(gt_binary).sum().cuda(0)
    FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
    FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda(0)
    # recall
    Recall = TP / (TP + FN)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # F1 score = Dice
    Dice = 2 * Precision * Recall / (Precision + Recall)
    # Overall accuracy
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU = TP / (TP + FP + FN)

    return MAE.data.cpu().numpy().squeeze(), \
           Recall.data.cpu().numpy().squeeze(), \
           Precision.data.cpu().numpy().squeeze(), \
           Accuracy.data.cpu().numpy().squeeze(), \
           Dice.data.cpu().numpy().squeeze(), \
           IoU.data.cpu().numpy().squeeze()

def intersectionAndUnion(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    # assert output.shape == target.shape
    # output = output.reshape(output.size).copy()
    # target = target.reshape(target.size)
    output[target == ignore_index] = ignore_index
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=5, start_epoch=0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if epoch <= start_epoch:
        return 0.0
    return consistency * sigmoid_rampup(epoch - start_epoch, consistency_rampup - start_epoch)
