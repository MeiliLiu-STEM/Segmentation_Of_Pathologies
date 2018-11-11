from __future__ import absolute_import, print_function

import os
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import nibabel
from processing import *
from n4bias import N4BiasFieldCorrection
# Creation Class to load data
class Data():
    def __init__(self, label_convert_source,label_convert_target):
        self.data_folder = 'E:/BRATS2017/'
        self.modality     = ['flair', 't1', 't1ce', 't2']
        self.with_ground_truth    = True
        self.label_convert_source = label_convert_source
        self.label_convert_target = label_convert_target
        self.data_names    = 'E:/BRATS2017/Data_names/train_names.txt'

        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
     # load image       
    def __load_image(self, patient_name, mod):
        patient_dir = os.path.join(self.data_folder, patient_name)
        image_names = os.listdir(patient_dir)
        volume_name = None
        for image_name in image_names:
            if(mod + '.' in image_name):
                volume_name = image_name
                break
        volume_name = os.path.join(patient_dir, volume_name)
        img = nibabel.load(volume_name)
        volume = img.get_data()
        return volume, volume_name
    # Load All images of training set    
    def get_data(self):
        with open(self.data_names) as f:
            content = f.readlines()
        self.patient_names = [x.strip() for x in content]
        image_patient = []
        X = []
        Y = []
        bbox  = []
        for i in range(len(self.patient_names)):
            volume_list = []
            volume_name_list = []
            for mod_idx in range(len(self.modality)):
                volume, volume_name = self.__load_image(self.patient_names[i], self.modality[mod_idx])
                if(mod_idx == 0):
                    # Get limit of bounding box to crop the imageto eliminate background
                    bbmin, bbmax = shape_delimiter(volume)
                volume = coup_delimiter(volume, bbmin, bbmax)
                # Applying Correction intensity
                volume = N4BiasFieldCorrection(volume)
                # Normalize Data subtract mean and devise on stadanrd deviation
                volume = normalize(volume)
                volume_list.append(volume)
                volume_name_list.append(volume_name)
            image_patient.append(volume_name_list)
            X.append(volume_list)
            bbox.append([bbmin, bbmax])
            if(self.with_ground_truth):
                label,label_name = self.__load_image(self.patient_names[i], 'seg')
                label = coup_delimiter(label, bbmin, bbmax)
                Y.append(label)
        print('Data loaded')
        self.image_names = image_patient
        self.data   = X
        self.label  = Y
        self.bbox   = bbox
    # Get Batch of images to train
    def next_batch(self,indice_start):
        batch_size = 5
        data_shape = [150, 240, 240, 4]
        label_shape = [150, 240, 240, 1]
        #Number of slice to take
        data_slice_number = 50
        label_slice_number = 50
        batch_sample_model   = ['center', 'no', 'no']
        direction= 'axial' # axial, sagittal, coronal

        data = []
        label = []
        for i in range(indice_start,min(batch_size+indice_start,len(self.patient_names))):
            self.patient_id = i
            data_volumes = [x for x in self.data[self.patient_id]]
            if(self.with_ground_truth):
                label_volumes = [self.label[self.patient_id]]

            # Convert classification multi class to classification binaire
            label_volumes[0] = transform_y(label_volumes[0], self.label_convert_source, self.label_convert_target)
            # Transform data to correpond view Axial Sagittal Coronal        
            transposed_volumes = transform_direction(data_volumes, direction)
            volume_shape = transposed_volumes[0].shape
            patch_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            patch_label_shape =[label_slice_number, label_shape[1], label_shape[2]]
            center_point = center_roi_rand(volume_shape, patch_label_shape, batch_sample_model)
            patch_data = []
            for moda in range(len(transposed_volumes)):
                patch = center_roi(transposed_volumes[moda],center_point,patch_data_shape,'data')
                patch_data.append(patch)
            patch_data = np.asarray(patch_data)
            data.append(patch_data)
            
            if(self.with_ground_truth):
                tranposed_label = transform_direction(label_volumes, direction)
                sub_label = center_roi(tranposed_label[0], center_point,patch_label_shape,'label')
                label.append([sub_label])
                    
        data = np.asarray(data, np.float32)
        label = np.asarray(label, np.int64)
        batch = {}
        # resize to become [batch_size,slice_number,width,height,Modality]
        batch['images']  = np.transpose(data,   [0, 2, 3, 4, 1])
        batch['labels']  = np.transpose(label,  [0, 2, 3, 4, 1])
        
        return batch
    
        
    #Getteur Number total of images
    def total_image(self):
        return len(self.data)
    #Getteur Data of image
    def image_data(self, i):
        return [self.data[i],  self.patient_names[i], self.image_names[i], self.bbox[i]]
