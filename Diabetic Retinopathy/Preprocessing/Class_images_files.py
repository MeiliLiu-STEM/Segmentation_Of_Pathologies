import tensorlayer as tl
import os, csv
import shutil

"""
Classification data with correspond label
"""

DR_data_path = "D:/train-resized-256/"
DR_data_save = "D:/train_split1/"
if not os.path.exists(DR_data_save):
    os.makedirs(DR_data_save)

DR_path_list = tl.files.load_file_list(path=DR_data_path, regx='.*.jpeg',printable=False)

train_label_path = "D:/trainLabels_master_256_v21.csv"
train_id_list = []
train_label_list = []
with open(train_label_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        train_id_list.append(content[2])
        train_label_list.append(float(content[1]))
        
train_name_list = [p for p in DR_path_list]
count=0
for i in train_id_list:
    if i in train_name_list:
        if not os.path.exists(os.path.join(DR_data_save,str(train_label_list[count]))):
            os.makedirs(os.path.join(DR_data_save,str(train_label_list[count])))
        shutil.move(os.path.join(DR_data_path, i),os.path.join(DR_data_save,str(train_label_list[count]),i))
        count+=1

