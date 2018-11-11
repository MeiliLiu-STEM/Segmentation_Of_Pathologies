import numpy as np
from scipy import ndimage
import tensorflow as tf
from Data import *
from processing import *
from WTNet import WTNet
from ENet import ENet
import SimpleITK as sitk


def prediction(images, shape_data, shape_label, session, predict, x):
    
    [D, H, W] = images[0].shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob = np.zeros([D, H, W, 2])
    sub_image_baches = []
    for center_slice in range(int(shape_label[0]/2), D + int(shape_label[0]/2), shape_label[0]):
        center_slice = min(center_slice, D - int(shape_label[0]/2))
        sub_image_bach = []
        for chn in range(4):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = center_roi(images[chn], temp_input_center, shape_data)
            sub_image_bach.append(sub_image)
        sub_image_bach = np.asanyarray(sub_image_bach, np.float32)
        sub_image_baches.append(sub_image_bach)
    total_batch = len(sub_image_baches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_baches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.random.normal(0, 1, size = [4] + shape_data))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch = session.run(predict, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(prob_mini_batch.shape[0]):
            center_slice = sub_label_idx*shape_label[0] + int(shape_label[0]/2)
            center_slice = min(center_slice, D - int(shape_label[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], 1]
            sub_prob = np.reshape(prob_mini_batch[batch_idx], shape_label + [2])
            temp_prob = set_roi_to_images(temp_prob, temp_input_center, sub_prob)
            sub_label_idx = sub_label_idx + 1
    return temp_prob 

save_folder = 'E:/results/'
label_convert_source = [0, 1, 2, 4]
label_convert_target = [0, 1, 1, 1]
data = Data(label_convert_source,label_convert_target)
data.get_data()
image_num = data.total_image()
print('data done')
    
global_variable = tf.global_variables()
session = tf.InteractiveSession()   
session.run(tf.global_variables_initializer())  
batch_size  = 5
        
data_shape_Wax = [batch_size, 20, 160, 160, 4]
x_Waxial = tf.placeholder(tf.float32, shape = data_shape_Wax) 
model_whole_axial = WTNet
model_whole_axial = model_whole_axial(num_classes = 2, name = 'WNet_ax')
y_pred_cls = model_whole_axial(x_Waxial)
y_prob_label_Wax = tf.nn.softmax(y_pred_cls)
    
var_whole_ax = [x for x in global_variable if x.name[0:8]=='WNet_ax/']
restore1ax = tf.train.Saver(var_whole_ax)
restore1ax.restore(session, "Models/WNet_Ax_20000.ckpt")


data_shape_Wsg = [batch_size, 20, 160, 160, 4]
x_Wsagittal = tf.placeholder(tf.float32, shape = data_shape_Wsg)          
model_whole_sagittal = WTNet
model_whole_sagittal = model_whole_sagittal(num_classes = 2, name = 'WNet_sg')
y_pred_cls = model_whole_sagittal(x_Wsagittal)
y_prob_label_Wsg = tf.nn.softmax(y_pred_cls)
    
var_whole_sg = [x for x in global_variable if x.name[0:8]=='WNet_sg/']
restore1sg = tf.train.Saver(var_whole_sg)
restore1sg.restore(session, "Models/WNet_Sg_20000.ckpt")     
            
data_shape_Wcr = [batch_size, 20, 160, 160, 4]
x_Wcoronal = tf.placeholder(tf.float32, shape = data_shape_Wcr)          
model_whole_coronal = WTNet
model_whole_coronal = model_whole_coronal(num_classes =2, name = 'WNet_cr')
y_pred_cls = model_whole_coronal(x_Wcoronal)
y_prob_label_Wcr = tf.nn.softmax(y_pred_cls)
    
var_whole_cr = [x for x in global_variable if x.name[0:8]=='WNet_sg/']
restore1cr = tf.train.Saver(var_whole_cr)
restore1cr.restore(session, "Models/WNet_Sg_20000.ckpt")
print("net1")
            
            
data_shape_Tax = [batch_size, 20, 120, 120, 4]
x_Taxial = tf.placeholder(tf.float32, shape = data_shape_Tax)          
model_tumor_axial = WTNet
model_tumor_axial = model_tumor_axial(num_classes =2, name = 'TNet_ax')
y_pred_cls = model_tumor_axial(x_Taxial)
y_prob_label_Tax = tf.nn.softmax(y_pred_cls)
    
var_tumor_ax = [x for x in global_variable if x.name[0:8]=='TNet_ax/']
restore2ax = tf.train.Saver(var_tumor_ax)
restore2ax.restore(session, "Models/TNet_Ax_20000.ckpt")


data_shape_Tsg = [batch_size, 20, 120, 120, 4]
x_Tsagittal = tf.placeholder(tf.float32, shape = data_shape_Tsg)          
model_tumor_sagittal = WTNet
model_tumor_sagittal = model_tumor_sagittal(num_classes =2, name = 'TNet_sg')
y_pred_cls = model_tumor_sagittal(x_Tsagittal)
y_prob_label_Tsg = tf.nn.softmax(y_pred_cls)
    
var_tumor_sg = [x for x in global_variable if x.name[0:8]=='TNet_sg/']
restore2sg = tf.train.Saver(var_tumor_sg)
restore2sg.restore(session, "Models/TNet_Sg_20000.ckpt") 
                

data_shape_Tcr = [batch_size, 20, 120, 120, 4]
x_Tcoronal = tf.placeholder(tf.float32, shape = data_shape_Tcr)          
model_tumor_coronal = WTNet
model_tumor_coronal = model_tumor_coronal(num_classes =2, name = 'TNet_cr')
y_pred_cls = model_tumor_coronal(x_Tcoronal)
y_prob_label_Tcr = tf.nn.softmax(y_pred_cls)
    
var_tumor_cr = [x for x in global_variable if x.name[0:8]=='TNet_cr/']
restore2cr = tf.train.Saver(var_tumor_cr)
restore2cr.restore(session, "Models/TNet_Cr_20000.ckpt") 
print("net2")

            
          
data_shape_Eax = [batch_size, 20, 96, 96, 4]
x_Eaxial = tf.placeholder(tf.float32, shape = data_shape_Eax)          
model_enhanc_axial = ENet
model_enhanc_axial = model_enhanc_axial(num_classes =2, name = 'ENet_ax')
y_pred_cls = model_enhanc_axial(x_Eaxial)
y_prob_label_Eax = tf.nn.softmax(y_pred_cls)
    
var_enhanc_ax = [x for x in global_variable if x.name[0:8]=='ENet_ax/']
restore3ax = tf.train.Saver(var_enhanc_ax)
restore3ax.restore(session, "Models/ENet_Ax_20000.ckpt") 


data_shape_Esg = [batch_size, 20, 96, 96, 4]
x_Esagittal = tf.placeholder(tf.float32, shape = data_shape_Esg)          
model_enhanc_sagittal = ENet
model_enhanc_sagittal = model_enhanc_sagittal(num_classes =2, name = 'ENet_sg')
y_pred_cls = model_enhanc_sagittal(x_Esagittal)
y_prob_label_Esg = tf.nn.softmax(y_pred_cls)
    
var_enhanc_sg = [x for x in global_variable if x.name[0:8]=='ENet_sg/']
restore3sg = tf.train.Saver(var_enhanc_sg)
restore3sg.restore(session, "Models/ENet_Sg_20000.ckpt")     
                

data_shape_Ecr = [batch_size, 20, 96, 96, 4]
x_Ecoronal = tf.placeholder(tf.float32, shape = data_shape_Ecr)          
model_enhanc_coronal = ENet
model_enhanc_coronal = model_enhanc_coronal(num_classes =2, name = 'ENet_sg')
y_pred_cls = model_enhanc_sagittal(x_Ecoronal)
y_prob_label_Ecr = tf.nn.softmax(y_pred_cls)
    
var_enhanc_cr = [x for x in global_variable if x.name[0:8]=='ENet_cr/']
restore3cr = tf.train.Saver(var_enhanc_cr)
restore3cr.restore(session, "Models/ENet_Cr_20000.ckpt") 
print("net3")

    
for i in range(image_num):
    [patient_images,  patient_name, patient_img_names, images_bbox] = dataloader.image_data_(i)
    # test of 1st network
    data_shapes  = [ 20, 160, 160]
    label_shapes = [ 20, 160, 160]
    prob1_ax = prediction(patient_images, data_shapes, label_shapes, session, y_prob_label_Wax, x_Waxial)
    sagittal_images = transform_direction(patient_images, 'sagittal')
    prob1_sg = prediction(sagittal_images, data_shapes, label_shapes, session, y_prob_label_Wsg, x_Wsagittal)
    prob1_sg = np.transpose(prob1_sg, [1,2,0,3])
    coronal_images = transform_direction(patient_images, 'coronal')
    prob1_cr = prediction(coronal_images, data_shapes, label_shapes, session, y_prob_label_Wcr, x_Wcoronal)
    prob1_cr = np.transpose(prob1_cr, [1,0,2,3])
    prob1 = (prob1_ax+prob1_sg+prob1_cr)/3
    pred1 =  np.asarray(np.argmax(prob1, axis = 3), np.uint16)

    # 5.2, test of 2nd network
    holes1 = ndimage.morphology.binary_closing(pred1, structure = ndimage.generate_binary_structure(3, 2))
    bbox1 = shape_delimiter(holes1)
    whole = [coup_delimiter(img, bbox1[0], bbox1[1]) for img in patient_images]
        
    data_shapes  = [ 20, 120, 120]
    label_shapes = [ 20, 120, 120]
    prob2_ax = prediction(whole, data_shapes, label_shapes, session, y_prob_label_Tax, x_Taxial)
    sagittal_images = transform_direction(whole, 'sagittal')
    prob2_sg = prediction(sagittal_images, data_shapes, label_shapes, session, y_prob_label_Tsg, x_Tsagittal)
    prob2_sg = np.transpose(prob2_sg, [1,2,0,3])
    coronal_images = transform_direction(whole, 'coronal')
    prob2_cr = prediction(coronal_images, data_shapes, label_shapes, session, y_prob_label_Tcr, x_Tcoronal)
    prob2_cr = np.transpose(prob2_cr, [1,0,2,3])
    prob2 = (prob2_ax+prob2_sg+prob2_cr)/3
    pred2 = np.asarray(np.argmax(prob2, axis = 3), np.uint16)
             
    holes2 = ndimage.morphology.binary_closing(pred2, structure = ndimage.generate_binary_structure(3, 2))
    bbox2 = shape_delimiter(holes2)
    core = [coup_delimiter(one_img, bbox2[0], bbox2[1]) for one_img in whole]

    data_shapes  = [ 20, 120, 120]
    label_shapes = [ 20, 120, 120]
    prob3_ax = prediction(core, data_shapes, label_shapes, session, y_prob_label_Eax, x_Eaxial)
    sagittal_images = transform_direction(core, 'sagittal')
    prob3_sg = prediction(sagittal_images, data_shapes, label_shapes, session, y_prob_label_Esg, x_Esagittal)
    prob3_sg = np.transpose(prob3_sg, [1,2,0,3])
    coronal_images = transform_direction(core, 'coronal')
    prob3_cr = prediction(coronal_images, data_shapes, label_shapes, session, y_prob_label_Ecr, x_Ecoronal)
    prob3_cr = np.transpose(prob3_cr, [1,0,2,3])
    prob3 = (prob3_ax+prob3_sg+prob3_cr)/3

    pred3 = np.asarray(np.argmax(prob3, axis = 3), np.uint16)
             
    # fuse results at 3 levels
    Roi = np.zeros_like(pred2)
    Roi = insert_roi(Roi, bbox2[0], bbox2[1], pred3)
    enhanc_label = np.zeros_like(pred1)
    enhanc_label = insert_roi(enhanc_label, bbox1[0], bbox1[1], Roi)

    core_label = np.zeros_like(pred1)
    core_label = insert_roi(core_label, bbox1[0], bbox1[1], pred2)

    whole_label = (pred1 + core_label + enhanc_label) > 0
    whole_label = ndimage.morphology.binary_closing(whole_label, structure = ndimage.generate_binary_structure(3, 2))
            
    core_label = (core_label + enhanc_label) > 0
    core_label = core_label * whole_label
    core_label = ndimage.morphology.binary_closing(core_label, structure = ndimage.generate_binary_structure(3, 2))

    enhanc_label = core_label * enhanc_label

    # convert label and save output
    out_label = whole_label * 2
    out_label[core_label>0] = 1
    out_label[enhanc_label>0] = 4
    out_label = np.asarray(out_label, np.int16)
    final_label = np.zeros([155,240,240], np.int16)
    final_label = insert_roi(final_label, images_bbox[0], images_bbox[1], out_label)
    img = sitk.GetImageFromArray(final_label)
    img_ref = sitk.ReadImage(patient_img_names[0])
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, save_folder+"/{0:}.nii.gz".format(patient_name))
session.close()