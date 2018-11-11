from sklearn.model_selection import train_test_split
import shutil
import os
import tensorlayer as tl

"""
Split data to training set and test set

"""
DR_data_save = "D:/train_split1/"

DR_path_list = tl.files.load_folder_list(path=DR_data_save)
for i in DR_path_list:
    if not os.path.exists(os.path.join(i,'test')):
        os.makedirs(os.path.join(i,'test'))
    name_list = tl.files.load_file_list(path=i, regx='.*.jpeg',printable=False)
    x_train,x_test = train_test_split(name_list, test_size=0.2, random_state=42)
    for j in x_test:
        shutil.move(os.path.join(i,j),os.path.join(i,'test',j))
      
DR_data_final_train = "D:/train_split1/train/"
DR_data_final_test = "D:/train_split1/test/"       
if not os.path.exists(DR_data_final_train):
    os.makedirs(DR_data_final_train)  
if not os.path.exists(DR_data_final_test):
    os.makedirs(DR_data_final_test)
count = 0
for i in DR_path_list:
    name = os.path.basename(DR_path_list[count])
    test_file = tl.files.load_folder_list(path=i)
    shutil.move(test_file[0],os.path.join(DR_data_final_test,name))
    count+=1
count = 0
for i in DR_path_list:
    name = os.path.basename(DR_path_list[count])
    shutil.move(i,os.path.join(DR_data_final_train,name))
    count+=1
