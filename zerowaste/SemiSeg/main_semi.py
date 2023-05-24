import os
import sys
import torch
import logging
import argparse
import warnings
import torch.nn as nn
import torch.optim as optim

from losses import get_loss
from datasets import get_dataset
from config import get_cfg_defaults
from lr_scheduler import LR_Scheduler
from utils import valid_model, test_model, get_model, train_model

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        # default="./expconfigs/exp_unet_base_1x.yaml",
                        # default="./expconfigs/exp_unet_base_2x.yaml",
                        # default="./expconfigs/exp_unet_semi_1x.yaml",
                        # default="./expconfigs/exp_unet_semi_2x.yaml",
                        # default="./expconfigs/exp_unet_semi_3x.yaml",
                        # default="./expconfigs/exp_unet_semi_4x.yaml",
                        default="./expconfigs/exp_unet_semi_5x.yaml",
                        # default="/zerowaste/SemiSeg/expconfigs/exp_unet_semi_R1.yaml",
                        help="config yaml path")
    
    parser.add_argument("--load", type=str,
                        default="",
                        help="path to model weight")
    
    parser.add_argument("--valid", action="store_true",
                        # default=True,
                        help="enable evaluation mode for validation")
    
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")
    
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model runing mode (train/valid/test)")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args

def setup_logging(args, cfg):
    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr), logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}_{args.mode}.log'),
        mode='a')]
    logging.basicConfig(format=head, style='{', 
                        level=logging.DEBUG, handlers=handlers)
    logging.info(f'\n\nStart with config {cfg}')
    logging.info(f'Command arguments {args}')

def main(args, cfg):
    logging.info(f"=========> {cfg.EXP} <=========")
    start_epoch = 0

    model = get_model(cfg)
    if cfg.MODEL.WEIGHT != "":
        weight = cfg.MODEL.WEIGHT
        model.load_state_dict(torch.load(weight)["state_dict"], strict=False)

    # Define Loss and Optimizer
    train_criterion = get_loss(cfg)
    valid_criterion = get_loss(cfg)

    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        for loss_name in train_criterion.keys():
            train_criterion[loss_name] = train_criterion[loss_name].cuda()
        for loss_name in valid_criterion.keys():
            valid_criterion[loss_name] = valid_criterion[loss_name].cuda()

    # #optimizer
    if cfg.OPT.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=cfg.OPT.BASE_LR,
                                weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "adam":
        optimizer = optim.Adam(params=model.parameters(),
                               lr=cfg.OPT.BASE_LR,
                               weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPTIMIZER == "sgd":
        optimizer = optim.SGD(params=model.parameters(),
                              lr=cfg.OPT.BASE_LR,
                              weight_decay=cfg.OPT.WEIGHT_DECAY)
    else:
        raise Exception('OPT.OPTIMIZER should in ["adamw", "adam", "sgd"]')

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    train_loader = get_dataset('train' if not cfg.DATA.USING_MINI_TRAIN else 'train_mini', 
                               cfg,
                               train_size=cfg.DATA.SIZE,
                               crop_size=cfg.DATA.CROP_SIZE,
                               weak_aug=cfg.DATA.USING_WEAK_AUG,
                               semi_training=cfg.DATA.SEMI_TRAINING,
                               add_adeptive_noise=cfg.DATA.ADD_ADEPTIVE_NOISE,
                               max_pixel_detla=cfg.DATA.MAX_PIXEL_DETLA,
                               sup_color_aug_probs=cfg.DATA.SUP_COLOR_AUG_PROBS,
                               sup_offset_aug_probs=cfg.DATA.SUP_OFFSET_AUG_PROBS,
                               )
    unlabeled_train_loader = get_dataset('unlabeled',
                               cfg,
                               train_size=cfg.DATA.SIZE,
                               crop_size=cfg.DATA.CROP_SIZE,
                               weak_aug=cfg.DATA.USING_WEAK_AUG,
                               semi_training=cfg.DATA.SEMI_TRAINING,
                               add_adeptive_noise=cfg.DATA.ADD_ADEPTIVE_NOISE,
                               focus_add_adeptive_noise=cfg.DATA.FOCUS_ADD_ADEPTIVE_NOISE,
                               max_pixel_detla=cfg.DATA.MAX_PIXEL_DETLA,
                               sup_color_aug_probs=cfg.DATA.SUP_COLOR_AUG_PROBS,
                               sup_offset_aug_probs=cfg.DATA.SUP_OFFSET_AUG_PROBS,
                               aug_offline_path=cfg.DATA.AUG_OFFLINE_PATH,
                               )    
    
    valid_loader = get_dataset('test', 
                               cfg, 
                               train_size=cfg.DATA.SIZE,
                               add_adeptive_noise=False,
                               focus_add_adeptive_noise=False,
                               max_pixel_detla=0.,
                               sup_color_aug_probs=0.,
                               sup_offset_aug_probs=0.,                               
                               )

    # Load scheduler
    scheduler = LR_Scheduler("cos", 
                             cfg.OPT.BASE_LR, 
                             cfg.TRAIN.EPOCHS, 
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPT.WARMUP_EPOCHS)

    if args.mode == "train":
        train_dict = {"base_unet": train_model}
        train_dict[cfg.TRAIN.METHOD](logging.info, 
                                     cfg, 
                                     model, 
                                     train_loader, 
                                     unlabeled_train_loader, 
                                     train_criterion,
                                     optimizer, 
                                     scheduler, 
                                     start_epoch, 
                                     valid_loader)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_criterion, valid_loader)
    else:
        test_model(logging.info, cfg, model, valid_loader, weight=cfg.MODEL.WEIGHT)

if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.freeze()

    for _dir in ["WEIGHTS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    setup_logging(args, cfg)
    # setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
