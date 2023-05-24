import os
import cv2
import time
import json
import glob
import numpy as np

def adjust_gamma(image, gamma=1.0):
    brighter_image = np.array(np.power((image / 255.), gamma[:,:, None]) * 255, dtype=np.uint8)
    return brighter_image

offsets = np.load("/zerowaste/SemiSeg/database/zerowaste/detla_offsets/detla_173.npy")
img = cv2.imread('/zerowaste/SemiSeg/01_frame_000285.PNG')
for _ in range(1):
    t1 = time.time()
    h, w = img.shape[:2]
    gamma = cv2.resize((offsets + 1.5) ** 1.5, (w, h))
    aug_img = adjust_gamma(img, gamma)
    print(time.time() - t1)
# hog_image = 255 * (features[1]/ features[1].max())
cv2.imwrite("aug_img.jpg", aug_img)
print(1)
