import os
import torch
import warnings
import numpy as np
from tqdm import tqdm
import logging
from .metrics import DiceScoreStorer, IoUStorer
from .utils import AverageMeter, save_checkpoint, evaluate_seg, intersectionAndUnion, SegMetric

warnings.filterwarnings("ignore")

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def train_model(_print, 
                cfg, 
                model, 
                train_loader, 
                unlabeled_train_loader, 
                criterion, 
                optimizer,
                scheduler, 
                start_epoch, 
                test_loader):
    assert cfg.MODEL.SEG_HEAD_LOSS in criterion
    best_iou = 0.
    seg_metric = SegMetric()
    seg_metric.set_nclass(cfg.DATA.SEG_CLASSES + 1)
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")
        losses = AverageMeter()
        # top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        
        model.train()
        train_loader_iter = iter(train_loader)
        unlabeled_train_loader_iter = iter(unlabeled_train_loader)
        
        if cfg.MODEL.SEG_LOSS_ON:
            steps = len(train_loader)
        elif cfg.MODEL.SEMI_SEG_LOSS_ON:
            steps = len(unlabeled_train_loader)
        else:
            raise TypeError
        for step in range(steps):
            losses_dict = {}
            batch_size = 1
            if cfg.MODEL.SEMI_SEG_LOSS_ON:
                unlabeled_batch = next(unlabeled_train_loader_iter)
                unlabeled_images = unlabeled_batch['image'].to(device='cuda', dtype=torch.float)
                unlabeled_wo_aug_images = unlabeled_batch['wo_aug_image'].to(device='cuda', dtype=torch.float)
                unlabeled_outputs = model(torch.cat((unlabeled_wo_aug_images, unlabeled_images)),
                                           add_feats_noise=cfg.MODEL.ADD_FEATS_NOISE,
                                           add_feats_drop=cfg.MODEL.ADD_FEATS_DROP,
                                           )
                unlabeled_wo_aug_outputs = unlabeled_outputs['seg_logits']
                unlabeled_w_aug_outputs = unlabeled_outputs['aug_seg_logits']
                semi_seg_loss = criterion[cfg.MODEL.SEMI_SEG_HEAD_LOSS](
                    unlabeled_w_aug_outputs, 
                    unlabeled_wo_aug_outputs,
                    "unsup_l1_loss")
                scaled_semi_seg_loss = {}
                for key in semi_seg_loss.keys():
                    scaled_semi_seg_loss[key] = semi_seg_loss[key] * \
                    cfg.MODEL.SEMI_SEG_LOSS_SCALE
                losses_dict.update(scaled_semi_seg_loss)
            
                if cfg.MODEL.ADD_SEMI_SEG_BCE_LOSS and \
                    epoch > cfg.MODEL.ADD_SEMI_SEG_BCE_LOSS_EPOCHS:
                    unlabeled_seg_outputs = model(
                        torch.cat((unlabeled_wo_aug_images, unlabeled_images)))                
                    unlabeled_seg_wo_aug_outputs, unlabeled_seg_w_aug_outputs = \
                        unlabeled_seg_outputs['seg_logits'].chunk(2)
                    semi_seg_bec_loss = criterion[cfg.MODEL.SEMI_SEG_HEAD_LOSS](
                        unlabeled_seg_w_aug_outputs, 
                        unlabeled_seg_wo_aug_outputs,
                        "unsup_ce_loss")
                    scaled_semi_seg_bce_loss = {}
                    for key in semi_seg_bec_loss.keys():
                        scaled_semi_seg_bce_loss[key] = semi_seg_bec_loss[key] * \
                        cfg.MODEL.SEMI_SEG_LOSS_SCALE
                    losses_dict.update(scaled_semi_seg_bce_loss)
                if not cfg.MODEL.SEG_LOSS_ON:
                    batch_size = unlabeled_images.size(0)
            mean_iou = 0.
            if cfg.MODEL.SEG_LOSS_ON:
                batch = next(train_loader_iter)
                labeled_images = batch['image'].to(device='cuda', dtype=torch.float)
                labeled_seg_targets = batch['seg_mask'].to(device='cuda', dtype=torch.float)
                labeled_outputs = model(labeled_images)
                labeled_output_preds = labeled_outputs["seg_logits"]
                seg_loss = criterion[cfg.MODEL.SEG_HEAD_LOSS](
                    labeled_output_preds, 
                    labeled_seg_targets,
                    "sup_ce_loss") 
                losses_dict.update(seg_loss)
                # if step % 5 == 0:
                seg_metric.reset()
                seg_metric.update(labels=labeled_seg_targets, pred_logits=labeled_output_preds)
                ious_lst, mean_iou = seg_metric.get_miou() 
                # else:
                #     ious_lst =[0., 0., 0., 0., 0.]                
                #     mean_iou = 0.               
                # top_dice.update(labeled_output_preds, labeled_seg_targets)
                # top_iou.update(labeled_output_preds, labeled_seg_targets)

            if len(losses_dict) < 1:
                continue
            
            # print(losses_dict)
            loss = sum(list(losses_dict.values()))
            loss.backward()
            scheduler(optimizer, step, epoch, None)
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss.item() * cfg.OPT.GD_STEPS, batch_size)
            # print_content = f"Epoch {epoch + 1} -> Train iou: %.3f, dice: %.3f, loss: %.5f " % (
            #     top_iou.avg, top_dice.avg, losses.avg)
            print_content = f"Epoch {epoch + 1} -> Train iou: %.3f, loss: %.5f " % (
                mean_iou, losses.avg)            
            for loss_name, loss_value in losses_dict.items():
                print_content += loss_name + " : %.5f " % (loss_value.item())
            _print(print_content)

        if (epoch + 1) % cfg.VAL.EPOCH == 0:
            ious_lst, mean_iou = eval_model(model, test_loader, cfg.DATA.SEG_CLASSES)
            _print("single class: {}".format(ious_lst))
            _print("mean iou: {}".format(mean_iou))
            # eval_model(_print, cfg, model, test_loader)
            # dice_test, iou_test_ = \
            #     test_model(_print, cfg, model, test_loader)
            iou_test = int(mean_iou * 10000) / 10000.
            _print("iou_test: {}".format(iou_test))
            if best_iou < iou_test  and iou_test > 0.30 and epoch > 10:
                best_iou = iou_test
                _print("best_iou: {}".format(best_iou))
                ckpt_path = os.path.join(cfg.DIRS.WEIGHTS, 
                                         cfg.EXP, 
                                         "epoch_{}_{}.pth".format(
                                             epoch + 1, iou_test))
                save_checkpoint({"epoch": epoch + 1,
                                 "arch": cfg.EXP,
                                 "state_dict": model.state_dict(),
                                 },
                                ckpt_path=ckpt_path)

def valid_model(_print, cfg, model, valid_criterion, valid_loader):
    losses = AverageMeter()
    top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    model.eval()
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, batch in enumerate(tbar):
            image = batch['image']
            seg_target = batch['seg_mask']
            image = image.to(device='cuda', dtype=torch.float)
            seg_target = seg_target.to(device='cuda', dtype=torch.float)
            output = model(image)
            output_target = output["seg_logits"]
            assert cfg.MODEL.SEG_HEAD_LOSS in valid_criterion
            loss = valid_criterion[cfg.MODEL.SEG_HEAD_LOSS](
                output_target, seg_target)
            top_dice.update(output_target, seg_target)
            top_iou.update(output_target, seg_target)

            # record
            losses.update(loss.item(), image.size(0))

    _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))

    return top_dice.avg, top_iou.avg

def test_model(_print, cfg, model, test_loader, weight=''):
    if weight != '':
        model.load_state_dict(torch.load(weight)["state_dict"])

    model.eval()
    tbar = tqdm(test_loader)
    MAE = []
    Recall = []
    Precision = []
    Accuracy = []
    Dice = []
    IoU = []

    with torch.no_grad():
        for batch in tbar:
            image = batch['image']
            seg_target = batch['seg_mask']

            image = image.to(device='cuda', dtype=torch.float)
            seg_target = seg_target.to(device='cuda', dtype=torch.float)
            output = model(image)["seg_logits"]
            out_evl = evaluate_seg(
                output.permute(0, 2, 3, 1).squeeze(), seg_target.squeeze())
            MAE.append(out_evl[0])
            Recall.append(out_evl[1])
            Precision.append(out_evl[2])
            Accuracy.append(out_evl[3])
            Dice.append(out_evl[4])
            IoU.append(out_evl[5])

    _print('=========================================')
    _print('MAE: %.3f' % np.mean(MAE))
    _print('Recall: %.3f' % np.mean(Recall))
    _print('Precision: %.3f' % np.mean(Precision))
    _print('Accuracy: %.3f' % np.mean(Accuracy))
    _print('Dice: %.3f' % np.mean(Dice))
    _print('IoU: %.3f' % np.mean(IoU))
    return np.mean(Dice), np.mean(IoU)

def eval_model(model, test_loader, num_classes=4, weight=''):
    if weight != '':
        model.load_state_dict(torch.load(weight)["state_dict"])
    model.eval()
    tbar = tqdm(test_loader)
    seg_metric = SegMetric()
    seg_metric.set_nclass(num_classes + 1)
    seg_metric.reset()
    with torch.no_grad():
        for batch in tbar:
            image = batch['image']
            seg_target = batch['seg_mask']
            image = image.to(device='cuda', dtype=torch.float)
            seg_target = seg_target.to(device='cuda', dtype=torch.float)
            output = model(image)["seg_logits"]
            seg_metric.update(labels=seg_target, pred_logits=output)
    ious_lst, mean_iou = seg_metric.get_miou()    
    return ious_lst, mean_iou
