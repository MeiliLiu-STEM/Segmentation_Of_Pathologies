import numpy as np
import pandas as pd
from PIL import Image

# Convert images to numpy array 

def convert_images_to_arrays_train(file_path, df):
    lst_imgs = [l for l in df['train_image_name']]
    print(len(lst_imgs))
    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])

labels = pd.read_csv("D:/trainLabels3.csv")

print("Writing Train Array")
X_train = convert_images_to_arrays_train('E:/data/preprocess/', labels)

print(X_train.shape)

print("Saving Train Array")
np.save('D:/Data2/X_train1.npy', X_train)