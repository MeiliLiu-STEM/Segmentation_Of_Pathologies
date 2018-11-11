import os
from skimage import io
from skimage.transform import resize
# Crop the images and resize to 256x256

def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    dirs = [l for l in os.listdir(path)]
    total = 0

    for item in dirs:
        img = io.imread(path+item)
        x,y,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[startx:startx+cropx,starty:starty+cropy]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)


crop_and_resize_images(path='E:/data/', new_path='E:/data/preprocess/', cropx=1800, cropy=1800, img_size=256)