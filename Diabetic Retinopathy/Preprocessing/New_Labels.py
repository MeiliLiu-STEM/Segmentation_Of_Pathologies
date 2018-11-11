import os
import pandas as pd

# Create new file labels contains all new images and affect each new image to original

trainLabels = pd.read_csv("D:/trainLabels2.csv")

lst_imgs = [i for i in os.listdir('E:/data/preprocess/')]

new_trainLabels = pd.DataFrame({'image': lst_imgs})
new_trainLabels['image2'] = new_trainLabels.image
# Remove the suffix from the image names.
new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

# Strip and add .jpeg back into file name
new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

new_trainLabels.columns = ['train_image_name', 'image']

trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
trainLabels.drop(['black'], axis=1, inplace=True)
trainLabels = trainLabels.dropna()
print(trainLabels.shape)

print("Writing CSV")
trainLabels.to_csv('D:/trainLabels3.csv', index=False, header=True)