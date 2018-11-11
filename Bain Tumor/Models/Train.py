from __future__ import absolute_import, print_function

import numpy as np
import random
import time
import tensorflow as tf
from niftynet.layer.loss_segmentation import LossFunction
from Data import Data
from WTNet import WTNet

random.seed(1)
# multiclass label
label_convert_source = [0, 1, 2, 4]
# binaire class
label_convert_target = [0, 1, 1, 1]
# Load data
data = Data(label_convert_source,label_convert_target)
data.get_data()
print('data done')    

model_name    = 'WNet'
class_num   = 2
batch_size  = 5
   
# Start building model
data_shape  = [batch_size,155, 240, 240, 4]
label_shape = [batch_size,155, 240, 240, 1]
# Initialisation of variable
x = tf.placeholder(tf.float32, shape = data_shape)
w = tf.placeholder(tf.float32, shape = label_shape)
y = tf.placeholder(tf.int64,   shape = label_shape)
# Regularizer to prevent Over-fitting
w_regularizer = tf.contrib.layers.python.layers.regularizers.l2_regularizer(1e-7)
b_regularizer = tf.contrib.layers.python.layers.regularizers.l2_regularizer(1e-7)
#load model
model = WTNet
model = model(num_classes = class_num,w_regularize=w_regularizer,b_regularizer=b_regularizer,name = model_name)
pred_label = model(x)
# Cost Function (1-Dice score)
loss_func = LossFunction(n_class=class_num)
loss = loss_func(pred_label, y, weight_map = w)
    
opt_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
session = tf.InteractiveSession()   
session.run(tf.global_variables_initializer())  
saver = tf.train.Saver()
    
# Start training
loss_file = "Models/WNet_wt_axial_loss.txt"
loss_list = []
print('start train')
for n in range(0, 20000):
    batch_dice_list = []
    for b in range(data.total_image()//batch_size + 1):
        batch = data.next_batch(b*batch_size)
        feed_x = batch['images']
        feed_y = batch['labels']
        print(n)
        _, dice = session.run([opt_step, loss], feed_dict ={x:feed_x, y:feed_y})
        batch_dice_list.append(dice)
    batch_dice = np.asarray(batch_dice_list, np.float32).mean()
    t = time.strftime('%X %x %Z')
    print(t, 'n', n,'loss', batch_dice)
    loss_list.append(batch_dice)
    # save loss values
    if(n%100 == 0):
        np.savetxt(loss_file, np.asarray(loss_list))
    #Save weights of model
    if((n+1)%5000  == 0):
        saver.save(session, "Models/WNet_Ax")
session.close()