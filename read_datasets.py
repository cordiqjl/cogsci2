import pandas as pd
import numpy as np
import os
import cv2


def load_fer2013(filename='fer2013/fer2013.csv'):
    df = pd.read_csv(filename, sep=',', header=0)
    images = []
    labels = []
    for each in df['pixels'].tolist():
        pom = np.array(each.split(' '))
        pom = np.reshape(pom, (48, 48))
        images.append(pom.astype(np.float))
    for each in df['emotion'].tolist():
        labels.append(int(each))
    return images, labels


def load_cohn_canade(rootdir='Cohn_canade/extended-cohn-kanade-images/',
                     rootdir2='Cohn_canade/Emotion_labels/Emotion'):
    f_img = []
    f_img_path = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            f_img_path.append(os.path.join(subdir, file))
            f_img.append(f_img_path[-1].split('/')[-1].split('.')[0])

    f_lbl = []
    f_lbl_path = []
    for subdir, dirs, files in os.walk(rootdir2):
        for file in files:
            f_lbl_path.append(os.path.join(subdir, file))
            f_lbl.append(f_lbl_path[-1].split('/')[-1].split('.')[0][:-8])

    label = []
    image = []
    for i in range(len(f_lbl)):
        for j in range(len(f_img)):
            if f_lbl[i] == f_img[j]:
                label.append(int(np.loadtxt(f_lbl_path[i])))
                image.append(cv2.imread(f_img_path[j], 0))
    return label, image
