#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Training script using custom coco format dataset
what you need to do is simply change the img_dir and annotation path here
Also define your own categories.

"""
import argparse
import sys
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    # default_argument_parser,
    launch,
)
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from train_det import Trainer, setup


def register_custom_datasets():
    # facemask dataset
    DATASET_ROOT = "./datasets/facemask"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
    VAL_PATH = os.path.join(DATASET_ROOT, "val")
    TRAIN_JSON = os.path.join(ANN_ROOT, "instances_train2017.json")
    VAL_JSON = os.path.join(ANN_ROOT, "instances_val2017.json")
    register_coco_instances("facemask_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("facemask_val", {}, VAL_JSON, VAL_PATH)

    # tl dataset
    DATASET_ROOT = "./datasets/tl"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    VAL_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    TRAIN_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_val_val.json")
    register_coco_instances("tl_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("tl_val", {}, VAL_JSON, VAL_PATH)

    # visdrone dataset
    DATASET_ROOT = "./datasets/visdrone"
    ANN_ROOT = os.path.join(DATASET_ROOT, "visdrone_coco_anno")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "VisDrone2019-DET-train/images")
    VAL_PATH = os.path.join(DATASET_ROOT, "VisDrone2019-DET-val/images")
    TRAIN_JSON = os.path.join(ANN_ROOT, "VisDrone2019-DET_train_coco.json")
    VAL_JSON = os.path.join(ANN_ROOT, "VisDrone2019-DET_val_coco.json")
    register_coco_instances("visdrone_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("visdrone_val", {}, VAL_JSON, VAL_PATH)

    # wearmask dataset
    DATASET_ROOT = "./datasets/wearmask"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "images/train2017")
    VAL_PATH = os.path.join(DATASET_ROOT, "images/val2017")
    TRAIN_JSON = os.path.join(ANN_ROOT, "train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "val.json")
    register_coco_instances("mask_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("mask_val", {}, VAL_JSON, VAL_PATH)

    # VOC dataset in coco format
    DATASET_ROOT = "./datasets/voc"
    ANN_ROOT = DATASET_ROOT
    TRAIN_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    VAL_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    TRAIN_JSON = os.path.join(ANN_ROOT, "annotations_coco_train_2012.json")
    VAL_JSON = os.path.join(ANN_ROOT, "annotations_coco_val_2012.json")

    register_coco_instances("voc_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("voc_val", {}, VAL_JSON, VAL_PATH)

    # zerawaste dataset
    DATASET_ROOT = "/root/autodl-tmp/zerowaste_database"
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train", "data")
    VAL_PATH = os.path.join(DATASET_ROOT, "test", "data")
    TRAIN_JSON = os.path.join(DATASET_ROOT, "train", "labels.json")
    VAL_JSON = os.path.join(DATASET_ROOT, "test", "labels.json")
    register_coco_instances("zerawaste_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("zerawaste_val", {}, VAL_JSON, VAL_PATH)

register_custom_datasets()

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file",
                        default="/zerowaste/third_party/yolov7_mask/configs/coco/darknet53_zerowaste.yaml", 
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
