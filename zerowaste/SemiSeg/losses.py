import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def get_loss(cfg):
    loss = {}
    for loss_name in cfg.MODEL.LOSSES:
        if loss_name == "BCEDiceLoss":
            loss[loss_name] = BCEDiceLoss()
        elif loss_name == "FocusBCEDiceLoss":
            loss[loss_name] = FocusBCEDiceLoss()
        elif loss_name == "OHMEBCEDiceLoss":
            loss[loss_name] = OHMEBCEDiceLoss()
        elif loss_name == "AdaptiveBCELoss":
            loss[loss_name] = AdaptiveBCELoss()
        elif loss_name == "SemiMixLoss":
            loss[loss_name] = SemiMixLoss()
        elif loss_name == "L1Loss":
            loss[loss_name] = L1Loss()
        elif loss_name == "BCLoss":
            loss[loss_name] = BCLoss()   
        elif loss_name == "AdaptiveBCLoss":
            loss[loss_name] = AdaptiveBCLoss()   
        elif loss_name == "SoftmaxL1Loss":
            loss[loss_name] = SoftmaxL1Loss()               
        else:
            raise TypeError("unsrpport loss type: {}".format(loss_name))            
    return loss

def dice_loss(preds, targets, smooth=1):
    size = preds.size(0)
    probs_flat = preds.view(size, -1)
    targets_flat = targets.view(size, -1)
    intersection = probs_flat * targets_flat
    dice_score = (2 * intersection.sum(1) + smooth) / \
                 (probs_flat.sum(1) + targets_flat.sum(1) + smooth)
    dice_loss_ = 1 - dice_score.sum() / size
    return dice_loss_

class L1Loss(nn.Module):
    def __init__(self, scale=1., l1_loss_thr=0.05):
        super(L1Loss, self).__init__()
        self.scale = scale
        self.l1_loss_thr = l1_loss_thr

    def forward(self, preds, targets_, loss_name="l1_loss"):
        probs = preds.sigmoid()
        targets = targets_.sigmoid()
        l1_loss = F.l1_loss(probs, targets, reduce=False)
        weights = (l1_loss > self.l1_loss_thr).float()
        loss_dict = {
            loss_name: ((weights * l1_loss).sum() / \
                (weights.sum() + 1.)) * self.scale
        }
        return loss_dict

class SoftmaxL1Loss(nn.Module):
    def __init__(self, scale=1., l1_loss_thr=0.05):
        super(SoftmaxL1Loss, self).__init__()
        self.scale = scale
        self.l1_loss_thr = l1_loss_thr

    def forward(self, preds, targets_, loss_name="l1_loss"):
        probs = torch.softmax(preds, dim=1)
        targets = torch.softmax(targets_, dim=1)
        l1_loss = F.l1_loss(probs, targets, reduce=False)
        weights = (l1_loss > self.l1_loss_thr).float()
        loss_dict = {
            loss_name: ((weights * l1_loss).sum() / \
                (weights.sum() + 1.)) * self.scale
        }
        return loss_dict

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets, masks=None, add_semi_loss=False):
        targets_ = targets[masks][:, None]
        preds_ = preds[masks]
        probs = preds_.sigmoid()
        weit = 1 + 5 * torch.abs(F.avg_pool2d(
            targets_, kernel_size=31, stride=1, padding=15) - targets_)
        wbce = F.binary_cross_entropy_with_logits(
            preds_, targets_, reduce=False)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        size = preds_.size(0)
        probs_flat = probs.view(size, -1)
        targets_flat = targets_.view(size, -1)
        intersection = probs_flat * targets_flat
        dice_score = (2 * intersection.sum(1) + self.smooth) / \
                     (probs_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice_score.sum() / size
        return (wbce + dice_loss).mean()

class AdaptiveBCEDiceLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=3., postive_thr=0.99, negative_thr=0.90, smooth=1):
        super(AdaptiveBCEDiceLoss, self).__init__()
        self.postive_thr = postive_thr
        self.negative_thr = negative_thr
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, masks=None, add_semi_loss=False):
        targets_ = targets[masks][:, None]
        preds_ = preds[masks]
        probs = preds_.sigmoid()
        postive_weights = torch.exp(self.beta - probs) * targets_
        negtive_weights = torch.exp(probs) * (1.0 - targets_)

        weights = postive_weights + negtive_weights
        bce_loss = weights * F.binary_cross_entropy_with_logits(
            preds_, targets_, reduce=False)
        bce_loss = bce_loss.mean()
        
        size = preds_.size(0)
        probs_flat = probs.view(size, -1)
        targets_flat = targets_.view(size, -1)
        intersection = probs_flat * targets_flat
        dice_score = (2 * intersection.sum(1) + self.smooth) / \
                     (probs_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        dice_loss = 1. - dice_score.sum() / size
        return bce_loss + dice_loss

class AdaptiveBCELoss(nn.Module):
    def __init__(self, 
                 beta=1.0, 
                 gamma=3., 
                 postive_thr=0.99, 
                 negative_thr=0.90, 
                 smooth=1):
        super(AdaptiveBCELoss, self).__init__()
        self.postive_thr = postive_thr
        self.negative_thr = negative_thr
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets_, loss_name="sup_bce_loss"):
        probs = preds.sigmoid()
        targets = targets_[:, None]
        postive_weights = torch.exp(self.beta - probs) * targets
        negtive_weights = torch.exp(probs) * (1.0 - targets)
        postive_masks = (probs < self.postive_thr) * targets
        negtive_masks = (1. - probs < self.negative_thr) * (1. - targets)
        weights = (postive_weights + negtive_weights) * \
            (postive_masks + negtive_masks)
        bce_loss = weights * F.binary_cross_entropy_with_logits(
            preds, targets, reduce=False)
        sup_bce_loss = bce_loss.sum() / (weights.sum() + 1.)
        loss_dict = {loss_name: sup_bce_loss}
        return loss_dict

class BCLoss(nn.Module):
    def __init__(self, 
                 beta=1.0, 
                 gamma=3., 
                 postive_thr=0.99, 
                 negative_thr=0.90, 
                 smooth=1):
        super(BCLoss, self).__init__()
        self.postive_thr = postive_thr
        self.negative_thr = negative_thr
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, loss_name="sup_bce_loss"):
        sup_ce_loss = F.cross_entropy(preds, targets.long(), reduce=False)
        # print(1)
        # postive_weights = torch.exp(self.beta - probs) * targets
        # negtive_weights = torch.exp(probs) * (1.0 - targets)
        # postive_masks = (probs < self.postive_thr) * targets
        # negtive_masks = (1. - probs < self.negative_thr) * (1. - targets)
        # weights = (postive_weights + negtive_weights) * \
        #     (postive_masks + negtive_masks)
        # bce_loss = weights * F.binary_cross_entropy_with_logits(
        #     preds, targets, reduce=False)
        # sup_bce_loss = bce_loss.sum() / (weights.sum() + 1.)
        loss_dict = {loss_name: sup_ce_loss.mean()}
        return loss_dict

class AdaptiveBCLoss(nn.Module):
    def __init__(self, 
                 beta=1.0, 
                 gamma=math.e, 
                 prob_thr=0.95, 
                 smooth=1,
                 category_weights=[1., 2., 1., 2., 1.],
                #  category_weights=[1., 1., 1., 1., 1.],
                 ):
        super(AdaptiveBCLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.prob_thr = prob_thr
        self.smooth = smooth
        self.category_weights = category_weights

    def forward(self, preds, targets, loss_name="sup_ce_loss"):
        probs = torch.softmax(preds, dim=1)
        with torch.no_grad():
            weights = torch.zeros_like(preds)
            target_probs = weights.scatter_(dim=1, 
                                            index=targets[:, None].long(), 
                                            src=probs)
            target_probs = target_probs.sum(dim=1)
            masks = (target_probs < self.prob_thr).float() 
            target_weights = torch.ones_like(target_probs)
            for cls_id, category_weight in enumerate(self.category_weights):
                target_weights[targets == cls_id] = category_weight          
            weights = masks * torch.pow(self.gamma, target_weights - target_probs)
        sup_ce_loss = weights * F.cross_entropy(preds, targets.long(), reduce=False)
        loss_dict = {loss_name: sup_ce_loss.sum() / (weights.sum() + 1.)}
        return loss_dict

class SemiMixLoss(nn.Module):
    def __init__(self, 
                 beta=1.0,
                 gamma=3., 
                 postive_thr=0.99, 
                 negative_thr=0.90, 
                 smooth=1,
                 pseudo_probs_thr=0.90, 
                 ema_alpha=0.99, 
                 semi_loss_scale=0.2,
                 ):
        super(SemiMixLoss, self).__init__()
        self.postive_thr = postive_thr #
        self.negative_thr = negative_thr #
        self.smooth = smooth #
        self.beta = beta
        self.gamma = gamma #
        self.ema_alpha = ema_alpha
        self.pseudo_probs_thr = pseudo_probs_thr
        self.semi_loss_scale = semi_loss_scale

    def forward(self, preds, targets, masks=None, add_semi_loss=False):
        labeld_preds_ = preds[masks]
        labeld_probs = labeld_preds_.sigmoid()
        labeld_targets_ = targets[masks][:, None]
        postive_weights = torch.exp(self.beta - labeld_probs) * labeld_targets_
        negtive_weights = torch.exp(labeld_probs) * (1.0 - labeld_targets_)
        postive_masks = (labeld_probs < self.postive_thr) * labeld_targets_
        negtive_masks = (1. - labeld_probs < self.negative_thr) * (1. - labeld_targets_)
        weights = (postive_weights + negtive_weights) * \
            (postive_masks + negtive_masks)
        labeled_bce_loss = weights * F.binary_cross_entropy_with_logits(
            labeld_preds_, labeld_targets_, reduce=False)
        sup_bce_loss = labeled_bce_loss.sum() / (weights.sum() + 1.)
        loss_dict = {"sup_bce_loss": sup_bce_loss}
        if add_semi_loss:
            labels_probs_mean = (labeld_probs * labeld_targets_).sum() / \
                (labeld_targets_.sum() + 1).item()
            pseudo_probs_thr = self.ema_alpha * self.pseudo_probs_thr + \
                (1. - self.ema_alpha) * labels_probs_mean
            
            # labels_probs_var = labeled_probs_flat.var()
            unlabled_masks = (~masks).to(torch.bool)
            unlabeld_preds_ = preds[unlabled_masks]
            unlabeld_probs = unlabeld_preds_.sigmoid()
            bulr_unlabeld_probs = F.avg_pool2d(
                input=unlabeld_probs, kernel_size=5, stride=1, padding=2)
            pos_pseudo_masks = (bulr_unlabeld_probs > pseudo_probs_thr) * \
                        (unlabeld_probs > pseudo_probs_thr)
            self.pseudo_probs_thr = pseudo_probs_thr.item()
            unlabeld_targets_ = pos_pseudo_masks.to(torch.float32)
            unlabeld_postive_masks = unlabeld_probs > \
            min(self.pseudo_probs_thr * 1.05, self.postive_thr)
            neg_bulr_unlabeld_probs = 1.- bulr_unlabeld_probs
            unlabeld_negtive_masks = \
                (self.pseudo_probs_thr * 1.05 < neg_bulr_unlabeld_probs) * \
                (neg_bulr_unlabeld_probs  < self.postive_thr * 1.05)
            unlabeld_weights = (unlabeld_postive_masks + unlabeld_negtive_masks)            
            unlabeled_bce_loss = unlabeld_weights * F.binary_cross_entropy_with_logits(
                unlabeld_preds_, unlabeld_targets_, reduce=False)
            unsup_bce_loss = self.semi_loss_scale * \
                unlabeled_bce_loss.sum() / (unlabeld_weights.sum() + 1.)
            loss_dict.update({"unsup_bce_loss": unsup_bce_loss})
        return loss_dict        

class FocusBCEDiceLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=3., postive_thr=0.99, negative_thr=0.90, smooth=1):
        super(FocusBCEDiceLoss, self).__init__()
        self.postive_thr = postive_thr
        self.negative_thr = negative_thr
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, masks=None, add_semi_loss=False):
        targets_ = targets[masks][:, None]
        preds_ = preds[masks]
        probs = preds_.sigmoid()
        postive_weights = torch.pow(1. - probs, self.gamma) * targets_
        negtive_weights = torch.pow(probs, self.gamma) * (1. - targets_)
        weights = postive_weights + negtive_weights
        bce_loss = weights * F.binary_cross_entropy_with_logits(
            preds_, targets_, reduce=False)
        bce_loss = bce_loss.sum() / weights.sum()
        size = preds_.size(0)
        probs_flat = probs.view(size, -1)
        targets_flat = targets_.view(size, -1)
        intersection = probs_flat * targets_flat
        dice_score = (2 * intersection.sum(1) + self.smooth) / \
                     (probs_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        dice_loss = 1. - dice_score.sum() / size
        return bce_loss + dice_loss

class OHMEBCEDiceLoss(nn.Module):
    def __init__(self, postive_thr=0.95, negative_thr=0.90, smooth=1):
        super(OHMEBCEDiceLoss, self).__init__()
        self.postive_thr = postive_thr
        self.negative_thr = negative_thr
        self.smooth = smooth

    def forward(self, preds, targets, masks=None, add_semi_loss=False):
        targets_ = targets[masks][:, None]
        preds_ = preds[masks]
        probs = preds_.sigmoid()
        postive_weights = (probs < self.postive_thr) * targets_
        negtive_weights = (1. - probs < self.negative_thr) * (1. - targets_)
        weights = postive_weights + negtive_weights
        bce_loss = weights * F.binary_cross_entropy_with_logits(
            preds_, targets_, reduce=False)
        bce_loss = bce_loss.sum() / weights.sum()
        size = preds_.size(0)
        probs_flat = probs.view(size, -1)
        targets_flat = targets_.view(size, -1)
        intersection = probs_flat * targets_flat
        dice_score = (2 * intersection.sum(1) + self.smooth) / \
                     (probs_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        dice_loss = 1. - dice_score.sum() / size
        return bce_loss + dice_loss

class OhemCELoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000, weight=None):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, seg_pred, seg_targets, **kwargs):
        pixel_losses = self.criterion(seg_pred, seg_targets).contiguous().view(-1)
        mask = seg_targets.contiguous().view(-1) != self.ignore_label
        tmp_target = seg_targets.clone()

        tmp_target[tmp_target == self.ignore_label] = 0
        # one_hot_label = F.one_hot(tmp_target)
        pred = F.softmax(seg_pred, dim=1)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()
    
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2., ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, seg_pred, seg_targets):
        scores = F.softmax(seg_pred, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(seg_pred, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, seg_targets)
        return loss
    
if __name__ == "__main__":
    loss = AdaptiveBCLoss()
    preds = torch.rand([1, 5, 224, 314])    
    targets = (torch.rand([1, 224, 314]) > 0.5).float()  
    cacl_loss = loss(preds, targets) 
    print(1)   