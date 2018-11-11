import time
import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorlayer as tl

## Detect the black images after crop images mean = 0 
def find_black_images(file_path, df):

    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]


if __name__ == '__main__':
    trainLabels = pd.read_csv('E:/data/trainLabels.csv')
    print(1)
    dirr = tl.files.load_file_list(path="E:/data/preprocess/", regx='.*.jpeg',printable=False)
    train_name_list = [os.path.splitext(p)[0] for p in dirr]

    train_all = trainLabels['image'].tolist()
    label_all = trainLabels['level'].tolist()
    liste_train = []
    liste_label = []
    count=0
    for i in train_all:
        if i in train_name_list:
            liste_train.append(i+'.jpeg')
            liste_label.append(label_all[count])
        count+=1
    trainLabels = pd.DataFrame({'image':liste_train,'level':liste_label})
    trainLabels['black'] = np.nan
    print(1)
    trainLabels['black'] = find_black_images('E:/data/preprocess/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels2.csv', index=False, header=True)

    print("Completed")