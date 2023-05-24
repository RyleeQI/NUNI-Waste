import cv2, os
import math
import torch
import random
import numpy as np
import glob
import shutil

def detla_maker(w, h):
    Dist = math.sqrt(w * h)
    coff = [
        (random.random() - 0.5) * 2. / 4.,
        (random.random() - 0.5) * 2.,
        (random.random() - 0.5) * 2. / 4.,
        (random.random() - 0.5) * 2.,
        (random.random() - 0.5) * 2.,
        (random.random() - 0.5) * 2.,
    ]
    point1 = int(w * coff[0])
    point2 = int(w * coff[1])

    point3 = int(Dist * coff[2])
    point6 = int(Dist * coff[3])
    point7 = int(Dist * coff[4])
    point10 = int(Dist * coff[5])

    point4 = point3 / 2.
    point5 = point6 / 2.

    point8 = point7 / 2.
    point9 = point10 / 2.
    fit_line = np.polyfit([point1, point2], [0, h - 1], 1)
    detla = []
    for y in range(h):
        g1 = (y - 1) * (point7 - point3) / (h - 1) + point3
        g2 = (y - 1) * (point8 - point4) / (h - 1) + point4
        g3 = 0
        g4 = (y - 1) * (point9 - point5) / (h - 1) + point5
        g5 = (y - 1) * (point10 - point6) / (h - 1) + point6
        y_values = [g1, g2, g3, g4, g5]
        x1 = 0
        x3 = int((y - fit_line[1])/fit_line[0])
        x5 = w - 1
        x2 = x3 // 2
        x4 = (x3 + x5) // 2
        x_values = [x1, x2, x3, x4, x5]
        fit_curve = np.polyfit(x_values, y_values, 3)
        per_detla = []
        for x in range(w):
            per_detla.append(fit_curve[0]*x**3 + fit_curve[1]*x**2 + fit_curve[2]*x + fit_curve[3])
        detla.append(per_detla)
    detla = np.array(detla)
    max_data = detla.max()
    min_data = detla.min()
    ratio = ((detla - min_data) / (max_data - min_data) - 0.5) * 2.
    return ratio

if __name__=="__main__":
    image_files = glob.glob(os.path.join("/InfraredSeg/testing", "blank_*.png"))
    names = [os.path.basename(image_file).split(".")[0] for image_file in image_files]
    src = "/InfraredSeg/database/zerowaste"
    for name in names:
        shutil.copy(os.path.join(src, "offsets", name.replace("blank", "detla") + ".npy"),
                                 os.path.join(src, "detla_offsets"))
    # for i in range(200):
    #     w = 640 + int(256 * (random.random() - 0.5) * 2.)
    #     h = 640 + int(256 * (random.random() - 0.5) * 2.)
    #     detla = detla_maker(w, h)
    #     np.save("offsets/detla_{}.npy".format(i), detla)
    #     blank = (detla * 128 + 128).clip(0, 255)
    #     cv2.imwrite("testing/blank_{}.png".format(i), blank)