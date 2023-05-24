import os
import cv2
import glob
import json
import random
import warnings

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from albumentations import ImageOnlyTransform

warnings.filterwarnings("ignore")

class InhomogenousColorAug(ImageOnlyTransform):
    def __init__(self, 
                 always_apply: bool = True, 
                 p: float = 1.0, 
                 max_pixel_detla=80.,
                 aug_offline_path="database"):
        self.detla_files = glob.glob(os.path.join(aug_offline_path, "*.npy"))
        self.detla_infos = {
            "detla_file": self.detla_files[0],
            "count": 0,
        }
        self.max_pixel_detla = max_pixel_detla
        self.count = 1
        super(InhomogenousColorAug, self).__init__(always_apply, p)
        
    def adjust_gamma(self, image, gamma=1.0):
        brighter_image = np.array(np.power((image / 255.), 
                                           gamma[:, :, None]) * 255, dtype=np.uint8)
        return brighter_image

    def apply(self, image, **params):
        if self.count % 10 == 0:
            idx = int(len(self.detla_files) * random.random())
            detla_file = self.detla_files[idx]
            self.detla_infos["detla_file"] = detla_file
            detla_offsets = np.load(self.detla_infos["detla_file"])
            self.detla_infos["detla_offsets"]= detla_offsets
        if "detla_offsets" in self.detla_infos:
            detla_offsets = self.detla_infos["detla_offsets"]
        else:
            detla_offsets = np.load(self.detla_infos["detla_file"])
            self.detla_infos["detla_offsets"]= detla_offsets
        # print("detla_file: ", self.detla_infos["detla_file"])            
        self.count += 1
        src_h, src_w = image.shape[:2]
        dh, dw = detla_offsets.shape
        cropped_h = dh * (random.randint(70, 100) / 100.)
        cropped_w = dw * (random.randint(70, 100) / 100.)
        
        y_pos = max(0, int(random.random() * (dh - cropped_h - 2)))
        ty = int(y_pos + cropped_h) 
        x_pos = max(0, int(random.random() * (dw - cropped_w - 2)))
        tx = int(x_pos + cropped_w) 
        crop_detla = detla_offsets[y_pos : ty, x_pos : tx]
        crop_detla = cv2.resize(crop_detla, (src_w, src_h))
        color_detla = crop_detla * self.max_pixel_detla
        if random.random() < 0.3:
            color_detla = -1. * color_detla
        # cv2.imwrite("detla.png", 255 * (color_detla - color_detla.min()) / (color_detla.max() - color_detla.min()))
        # if random.random() < 0.5: 
        #     color_detla = np.rot90(color_detla)
        if random.random() < 0.5:
            if random.random() < 0.5:
                color_detla = color_detla[::-1, :]
            else:
                color_detla = color_detla[:, ::-1]
        return self.adjust_gamma(image, (color_detla + 1.5) ** 1.2)
        # image_tmp = image.copy()
        # image_tmp = np.transpose(image.copy(), (2, 0, 1)).astype(np.float32)
        # img_aug = image.astype(np.float32) + color_detla[:, :, None]
        # return img_aug.clip(0, 255).astype(np.uint8)
        # return np.transpose(img_aug.clip(0, 255), (1, 2, 0)).astype(np.uint8)
    
class InhomogenousOffsetAug(ImageOnlyTransform):
    def __init__(self, 
                 always_apply: bool = True, 
                 p: float = 1.0, 
                 target_size=256, 
                 offset_val=80., 
                 curve_period=1200.):
        # self.target_size = target_size
        self.offset_val = offset_val
        self.curve_period = curve_period
        self.line_length = 1200
        # self.detla_x = np.arange(0, self.target_size)[None, :].repeat(self.target_size, 0).astype(np.float32)
        # self.detla_y = np.arange(0, self.target_size)[:, None].repeat(self.target_size, 1).astype(np.float32)        
        super(InhomogenousOffsetAug, self).__init__(always_apply, p)

    def apply(self, image, **params):
        src_h, src_w = image.shape[:2]
        detla_x = np.arange(0, src_w)[None, :].repeat(src_h, 0).astype(np.float32)
        detla_y = np.arange(0, src_h)[:, None].repeat(src_w, 1).astype(np.float32)        
        offset_val = (self.offset_val + 15. * (random.random() - 0.5) * 2.)
        curve_period = (self.curve_period + 200. * (random.random() - 0.5) * 2.)
        offset = offset_val * np.sin(2 * np.pi * \
            np.arange(0, self.line_length) / curve_period)
        if random.random() < 0.5:
            pose_idx = max(0, int((self.line_length - src_w - 2) * random.random()))
            cropped_offset = offset[pose_idx : pose_idx + src_w]
            crop_detla_x = detla_x + cropped_offset
            crop_detla_y = detla_y
        else:
            pose_idx = max(0, int((self.line_length - src_h - 2) * random.random()))
            cropped_offset = offset[pose_idx : pose_idx + src_h]            
            crop_detla_x = detla_x
            crop_detla_y = detla_y + cropped_offset[:, None]
        image_aug = cv2.remap(image.astype(np.float32),
                              crop_detla_x.astype(np.float32),
                              crop_detla_y.astype(np.float32),
                              cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_DEFAULT,
                              borderValue=0).astype(np.uint8)   
        return image_aug
        
class ImageDataset(Dataset):
    def __init__(self, 
                 cfg, 
                 roots, 
                 mode='train', 
                 train_size=(544, 960), 
                 crop_size=(512, 512),
                 weak_aug=False,
                 scale=(0.99, 1.01),
                 semi_training=False,
                add_adeptive_noise=False,
                focus_add_adeptive_noise=False,
                max_pixel_detla=90.,
                sup_color_aug_probs=0.1,
                sup_offset_aug_probs=0.2,                 
                aug_offline_path="./database/zerowaste/detla_offsets",                 
                 ):
        self.cfg = cfg
        self.mode = mode
        self.scale = scale
        self.semi_training = semi_training
        self.crop_size = crop_size
        self.train_size = train_size
        self.add_adeptive_noise = add_adeptive_noise
        self.focus_add_adeptive_noise = focus_add_adeptive_noise
        self.max_pixel_detla = max_pixel_detla
        self.sup_color_aug_probs = sup_color_aug_probs
        self.sup_offset_aug_probs = sup_offset_aug_probs
        self.aug_offline_path = aug_offline_path
        self.gts = []
        self.images = []
        self.dataset_lens = []
        self.base_path = "/root/autodl-tmp/zerowaste_database_resized"
        for root in roots:
            if mode == 'train':
                with open(os.path.join(root, "{}.json".format(mode)), "r") as f:
                    train_images = json.load(f)
                random.shuffle(train_images)
                _images = sorted(
                    [os.path.join(self.base_path, mode, "data", train_image) for train_image in train_images])
                _gts = sorted([_image.replace("/data/", "/sem_seg/") for _image in _images])
                if weak_aug:
                    self.transform = self.get_weak_augmentation()
                else:
                    self.transform = self.get_augmentation()
            elif mode == 'test':
                with open(os.path.join(root, "{}.json".format(mode)), "r") as f:
                    test_images = json.load(f)
                _images = sorted([os.path.join(self.base_path, mode, "data", test_image) for test_image in test_images])
                _gts = sorted([_image.replace("/data/", "/sem_seg/") for _image in _images])
                self.transform = A.Compose([
                    A.Resize(self.train_size[0], self.train_size[1], interpolation=cv2.INTER_NEAREST),  
                    # A.Resize(train_size, train_size, interpolation=cv2.INTER_NEAREST), 
                    ])
            elif mode == 'unlabeled':
                with open(os.path.join(root, "train_mini_{}.json".format(mode)), 'r') as f:
                    unlabeled_train_images = json.load(f)
                _images = sorted(
                    [os.path.join(self.base_path, "zerowaste-s-parts", "data", unlabeled_train_image) for unlabeled_train_image in unlabeled_train_images])
                _gts = sorted([_image for _image in _images])
                # _gts = sorted([_image.replace("/images/", "/masks/") for _image in _images])
                if weak_aug:
                    self.transform = self.get_weak_augmentation()
                else:
                    self.transform = self.get_augmentation()
            else:
                raise KeyError('MODE ERROR: {}'.format(mode))
            
            self.images += _images
            self.gts += _gts
            self.dataset_lens.append(len(self.images))
        # self.filter_files()
        self.size = len(self.images)
        self.to_tensors = A.Compose([A.Normalize(), ToTensorV2()])
        self.offset_aug = A.Compose([InhomogenousOffsetAug(
            offset_val=cfg.DATA.OFFSET_VALUE,
            curve_period=cfg.DATA.CURVE_PERIOD,
            )])
        self.color_aug = A.Compose([
            InhomogenousColorAug(p=1.0, 
                                 max_pixel_detla=max_pixel_detla, 
                                 aug_offline_path=aug_offline_path),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            # A.RandomBrightnessContrast(p=0.2),            
            ])

    def __len__(self):
        return self.size

    def lens(self):
        return self.dataset_lens

    def __getitem__(self, index):
        src_image = cv2.imread(self.images[index])
        image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        seg_mask = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        # seg_mask[seg_mask < 128] = 0
        # seg_mask[seg_mask >= 128] = 1
        # assert seg_mask.max() == 1 or seg_mask.max() == 0
        data_np = self.transform(image=image, mask=seg_mask)
        wo_aug_image = data_np["image"]
        wo_aug_mask = data_np["mask"]
        if random.random() < self.sup_offset_aug_probs:
            # cv2.imwrite("aug_f/{}_src1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_image)
            # cv2.imwrite("aug_f/{}_msk1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_mask*255)
            cat_aug_image = self.offset_aug(
                image=np.concatenate((wo_aug_image, wo_aug_mask[:, :, None]), -1))["image"]
            wo_aug_image = cat_aug_image[:, :, :3]
            wo_aug_mask = cat_aug_image[:, :, -1]
            # cv2.imwrite("aug_f/{}_src2.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_image)
            # cv2.imwrite("aug_f/{}_msk2.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_mask*255)
        if self.focus_add_adeptive_noise:
            aug_image = self.color_aug(image=wo_aug_image)["image"]
            # cv2.imwrite("aug_c/{}_src1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_image)
            # cv2.imwrite("aug_c/{}_src2.jpg".format(os.path.basename(self.images[index])[:-4]), aug_image)
            # cv2.imwrite("aug_c/{}_msk1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_mask*255)                
        else:
            if not self.add_adeptive_noise:
                aug_image = wo_aug_image
            else:
                if random.random() < self.sup_color_aug_probs:
                    aug_image = self.color_aug(image=wo_aug_image)["image"]
                    # cv2.imwrite("aug_c/{}_src1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_image)
                    # cv2.imwrite("aug_c/{}_src2.jpg".format(os.path.basename(self.images[index])[:-4]), aug_image)
                    # cv2.imwrite("aug_c/{}_msk1.jpg".format(os.path.basename(self.images[index])[:-4]), wo_aug_mask*255)
                else:
                    aug_image = wo_aug_image
                            
        data_tensor = self.to_tensors(
            image=aug_image,
            mask=wo_aug_mask,
        )
        wo_aug_data_tensor = self.to_tensors(
            image=wo_aug_image,
        )
        data = {'imidx': index, 
                'path': self.images[index], 
                'image': data_tensor['image'], 
                'wo_aug_image': wo_aug_data_tensor['image'], 
                'seg_mask': data_tensor['mask'],
        }
        return data

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        for img_path, gt_path in zip(self.images, self.gts):
            img = cv2.imread(img_path)
            gt = cv2.imread(gt_path)
            # assert gt.max() == 255
            assert gt.min() == 0
            # assert img.shape == gt.shape
            assert img_path.split('/')[-1].split('.')[0].split('_')[0] == \
                   gt_path.split('/')[-1].split('.')[0].split('_')[0], (img_path, gt_path)

    def get_augmentation(self):
        return A.Compose([
            A.Resize(self.train_size + 64, self.train_size + 64, 
                     interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomSizedCrop(min_max_height=(self.crop_size, self.train_size + 32),
                              height=self.train_size, width=self.train_size,
                              w2h_ratio=1.0, interpolation=cv2.INTER_NEAREST, p=0.8),
            A.Resize(self.train_size, self.train_size, interpolation=cv2.INTER_NEAREST)
        ])

    def get_weak_augmentation(self):
        return A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            A.Resize(self.train_size[0], self.train_size[1], 
                     interpolation=cv2.INTER_NEAREST),  
            A.RandomCrop(self.crop_size[0], self.crop_size[1]),          
        ])

def get_dataset(mode, 
                cfg, 
                train_size=(544, 960),
                crop_size=512, 
                scale=(0.75, 1), 
                weak_aug=False,
                semi_training=False,
                add_adeptive_noise=False,
                focus_add_adeptive_noise=False,
                max_pixel_detla=40.,
                sup_color_aug_probs=0.1,
                sup_offset_aug_probs=0.1, 
                aug_offline_path="./database/zerowaste/detla_offsets",               
                ):
    data_root = []
    if "zerowaste" in cfg.DATA.NAME:
        data_root.append(os.path.join(cfg.DIRS.DATA, 'zerowaste'))
        
    if mode == 'train':
        dts = ImageDataset(cfg=cfg, 
                           roots=data_root, 
                           mode=mode,
                           train_size=train_size, 
                           crop_size=crop_size, 
                           scale=scale,
                           weak_aug=weak_aug,
                           semi_training=semi_training,
                            add_adeptive_noise=add_adeptive_noise,
                            max_pixel_detla=max_pixel_detla,
                            sup_color_aug_probs=sup_color_aug_probs,
                            sup_offset_aug_probs=sup_offset_aug_probs,
                            aug_offline_path=aug_offline_path,
                           )
        dataloader = DataLoader(dts, 
                                shuffle=True,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                pin_memory=True, 
                                drop_last=True,
                                # worker_init_fn=worker_init_fn,
                                )
    elif mode == 'train_mini':
        dts = ImageDataset(cfg=cfg, 
                           roots=data_root, 
                           mode=mode, 
                           train_size=train_size,
                           crop_size=crop_size, 
                           scale=scale,
                           weak_aug=weak_aug,
                           semi_training=semi_training,
                            add_adeptive_noise=add_adeptive_noise,
                            max_pixel_detla=max_pixel_detla,
                            sup_color_aug_probs=sup_color_aug_probs,
                            sup_offset_aug_probs=sup_offset_aug_probs,
                            aug_offline_path=aug_offline_path,
                           )
        dataloader = DataLoader(dts, 
                                shuffle=True,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                pin_memory=True, 
                                drop_last=True,
                                # worker_init_fn=worker_init_fn
                                )
    elif mode == 'unlabeled':
        dts = ImageDataset(cfg=cfg, 
                           roots=data_root, 
                           mode=mode, 
                           train_size=train_size,
                           crop_size=crop_size, 
                           scale=scale,
                           weak_aug=weak_aug,
                           semi_training=semi_training,
                            add_adeptive_noise=add_adeptive_noise,
                            focus_add_adeptive_noise=focus_add_adeptive_noise,
                            max_pixel_detla=max_pixel_detla,
                            sup_color_aug_probs=sup_color_aug_probs,
                            sup_offset_aug_probs=sup_offset_aug_probs,
                            aug_offline_path=aug_offline_path,
                           )
        dataloader = DataLoader(dts, 
                                shuffle=True,
                                batch_size=cfg.TRAIN.UNLABELED_BATCH_SIZE,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                pin_memory=True, 
                                drop_last=True,
                                # worker_init_fn=worker_init_fn
                                )        
    elif mode == 'valid':
        dts = ImageDataset(cfg=cfg, 
                           roots=data_root, 
                           mode='val', 
                           train_size=train_size, 
                           scale=scale,
                            add_adeptive_noise=False,
                            max_pixel_detla=0.,
                            sup_color_aug_probs=0.,
                            sup_offset_aug_probs=0.,                           
                            aug_offline_path=aug_offline_path,
                           )
        dataloader = DataLoader(dts, 
                                batch_size=cfg.VAL.BATCH_SIZE,
                                shuffle=False, 
                                drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'test':
        dts = ImageDataset(cfg=cfg, 
                           roots=data_root, 
                           mode=mode, 
                           train_size=train_size, 
                           scale=scale,
                            add_adeptive_noise=False,
                            max_pixel_detla=0.,
                            sup_color_aug_probs=0.,
                            sup_offset_aug_probs=0.,                             
                            aug_offline_path=aug_offline_path,
                           )
        dataloader = DataLoader(dts, 
                                batch_size=cfg.TEST.BATCH_SIZE,
                                shuffle=False, 
                                drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    else:
        raise KeyError(f"mode error: {mode}")
    return dataloader
