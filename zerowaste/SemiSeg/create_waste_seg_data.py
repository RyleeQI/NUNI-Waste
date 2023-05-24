import os
import cv2,json
import numpy as np
import random
base_path = "/root/autodl-tmp/zerowaste_database_resized"
test_files = os.listdir(os.path.join(base_path, "zerowaste-s-parts", "data"))
random.shuffle(test_files)
with open("/zerowaste/SemiSeg/database/zerowaste/train_unlabeled.json", "w") as f:
    json.dump(test_files, f, indent=2)

# for mask_file in os.listdir(os.path.join(base_path, "train", "sem_seg")):
#     print(mask_file)
#     mask = cv2.imread(os.path.join(base_path, "train", "sem_seg", mask_file), cv2.IMREAD_GRAYSCALE)
#     mask[mask == 0] = 255
#     # mask[mask == 3] = 80
#     cv2.imwrite("mask.png", mask)
#     print(1)