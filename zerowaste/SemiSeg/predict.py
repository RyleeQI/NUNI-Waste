import os
import cv2
import time
import glob
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_cfg_defaults
from utils.models import unet_2D

def process_mask(y_pred):
    y_pred = y_pred.sigmoid()
    y_pred = y_pred.squeeze().cpu().numpy()
    y_pred = y_pred * 255.
    return y_pred

if __name__ == "__main__":
    time_taken = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg_defaults()
    cfg.merge_from_file("/root/MedSemiSeg/BoundarySemiSeg/expconfigs/exp_kvasir_boundary_unet_semi_test.yaml")
    cfg.freeze()

    model = unet_2D(method=cfg.MODEL.METHOD, cfg=cfg)
    # model = nn.DataParallel(model)
    # Load the saved model checkpoint
    pth_path = "/root/MedSemiSeg/BoundarySemiSeg/weights/semi_10/epoch_102_best_0.795799970626831.pth"
    ckpt = torch.load(pth_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], True)
    model.eval()
    model = model.to(device)

    # Set up the test folder and output folder
    src = "/root/MedSemiSeg/database/kvasir_seg/valid"
    save_path = "/root/MedSemiSeg/BoundarySemiSeg/pred_results"
    os.makedirs(save_path, exist_ok=True)

    # Get a list of image files in the test folder
    image_files = glob.glob(os.path.join(src, "*.jpg"))
    transform = A.Compose([A.Resize(320, 320, interpolation=cv2.INTER_NEAREST)])
    to_tensors = A.Compose([A.Normalize(), ToTensorV2()])

    for image_file in image_files:
        src_image = cv2.imread(image_file)
        _image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(src_image, (320, 320))
        image_dict = transform(image=_image)
        data_t = to_tensors(image=image_dict["image"])
        image_tensor = data_t["image"][None, ...]
        print(image_file)
        with torch.no_grad():
            start_time = time.time()
            output = model(image_tensor)["seg_logits"]
            end_time = time.time() - start_time
            time_taken.append(end_time)
            preds = process_mask(output).astype(np.uint8)
            preds = np.hstack((resized_image, 
                               np.concatenate([preds[:,:,None], preds[:,:,None], preds[:,:,None]],axis=-1)))
        cv2.imwrite(os.path.join(save_path, os.path.basename(image_file)), preds)
