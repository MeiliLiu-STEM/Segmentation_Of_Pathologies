import tensorflow as tf
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer import layer_util
from niftynet.layer.elementwise import ElementwiseLayer

# Class of Enhance Tumor
class ENet(TrainableLayer):
    def __init__(self,num_classes,acti_func='selu',w_regularizer = None,b_regularize=None,name='ENet'):

        super(ENet, self).__init__(name=name)
        self.b_regularizer = b_regularizer
        self.w_regularizer = w_regularizer
        self.num_classes = num_classes
        self.acti_func = acti_func

    def layer_op(self, images, is_training):
        #Creation Block of residual network
        with tf.name_scope("groupe1"):
            block1_1 = Residual(48,kernels=[1, 3, 3],acti_func=self.acti_func,name='block1_1')
            groupe1 = block1_1(images)
            block1_2 = Residual(48, kernels=[1, 3, 3],acti_func=self.acti_func,name='block1_2')
            groupe1 = block1_2(groupe1)
            layer1 = Layer(48,kernel_size=[3, 1, 1],padding='VALID',name='layer1')
            groupe1 = layer1(groupe1)
        
        with tf.name_scope("groupe2"):
            block2_1 = Residual(48,kernels=[1, 3, 3],acti_func=self.acti_func,name='block2_1')
            groupe1 = block2_1(groupe1)
            block2_2 = Residual(48,kernels=[1, 3, 3],acti_func=self.acti_func,name='block2_2')
            groupe1 = block2_2(groupe1)
            layer2 = Layer(48,kernel_size=[3, 1, 1],padding='VALID',name='layer2')
            groupe1 = layer2(groupe1)
        # Prediction level    
        with tf.name_scope("multi_scale1"):
            pred_scale1 = extract_tensor(groupe1,2)
            scale1 = Multi_scale(2,kernel_size=[1, 3, 3],padding='SAME',name='scale1')
            pred_scale1 = scale1(pred_scale1)
            
        with tf.name_scope("groupe3"):
           downsample = Layer(48,kernel_size=[1, 3, 3],stride=[1, 2, 2],padding='SAME',name='downsample') 
           groupe2 = downsample(groupe1)
           block3_1 = Residual(48,kernels=[1, 3, 3],dilation_rates=[1, 1, 1], acti_func=self.acti_func,name='block3_1')
           groupe2 = block3_1(groupe2)
           block3_2 = Residual(48,kernels= [1, 3, 3],dilation_rates=[1, 2, 2],acti_func=self.acti_func,name='block3_2')
           groupe2 = block3_2(groupe2)
           layer3 = Layer(48,kernel_size=[3, 1, 1],padding='VALID',name='layer3')
           groupe2 = layer3(groupe2)
           
        with tf.name_scope("groupe4"):
            block4_1 = Residual(48, kernels=[1, 3, 3],dilation_rates=[1, 3, 3], acti_func=self.acti_func,name='block4_1')
            groupe2 = block4_1(groupe2)
            block4_2 = Residual(48, kernels=[1, 3, 3], dilation_rates=[1, 3, 3],acti_func=self.acti_func,name='block4_2')
            groupe2 = block4_2(groupe2)
            layer4 = Layer(48,kernel_size=[3, 1, 1],padding='VALID',name='layer4')
            groupe2 = layer4(groupe2)
            
        with tf.name_scope("multi_scale2"):
            pred_scale2 = extract_tensor(groupe2,2)
            scale2 = Multi_scale(2,kernel_size=[1, 3, 3],stride=[1, 2, 2],padding='SAME',name='pred2')
            pred_scale2 = scale1(scale2)
            
        with tf.name_scope("groupe5"):
            block5_1 = Residual(48,kernels=[1, 3, 3],dilation_rates=[1, 2, 2],acti_func=self.acti_func,name='block5_1')
            groupe3 = block5_1(groupe2)
            block5_2 = Residual(48,kernels=[1, 3, 3],dilation_rates=[1, 1, 1],acti_func=self.acti_func,name='block5_2')
            groupe3 = block5_2(groupe2)
            layer5 = Layer(48,kernel_size=[3, 1, 1],padding='VALID',name='layer5')
            groupe3 = layer5(groupe3)
            
        with tf.name_scope("multi_scale3"):
            scale3 = Multi_scale(2,kernel_size=[1, 3, 3],stride=[1, 2, 2],padding='SAME',name='pred3')
            pred_scale3 = scale1(scale3)
            
        
        with tf.name_scope("final_scale"):
           scale4 = Multi_scale(2,kernel_size=[1, 3, 3],padding='SAME',name='final_scale')
           final_scale = scale4(tf.concat([scale1, scale2, scale3], axis=4, name='concate'))
        
        return final_scale

# Class of residual block
class Residual(TrainableLayer):
    def __init__(self,n_output_chns,kernels,strides, dilation_rates,acti_func,name):
        super(Residual, self).__init__(name=name)
        self.n_output_chns = n_output_chns
        self.kernels = kernels
        self.strides = strides
        self.dilation_rates = dilation_rates
        self.acti_func = acti_func
    # Activation Function Selu
    def selu(x, name):
        alpha = 1.6732
        scale = 1.0507
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    # Operation of layer
    def layer_op(self, tensor):
        output = tensor
        for i in range(2*len(self.kernels)):
            input_shape = tensor.shape.as_list()
            n_input_chns = input_shape[-1]
            spatial_rank = layer_util.infer_spatial_rank(tensor)
            w_full_size = layer_util.expand_spatial_params(self.kernel_size, spatial_rank)
            w_full_size = w_full_size + (n_input_chns, self.n_output_chns)
            conv_kernel = tf.get_variable('w', shape=w_full_size,
            regularizer=self.regularizers['w'])
            
            output = tf.layers.batch_normalization(input=tensor,name = 'bn_{}'.format(i))
            output = self.selu(output,name='acti_{}'.format(i))
            output = tf.nn.convolution(input=output,
                                          filter=conv_kernel,
                                          strides=self.strides,
                                          dilation_rate=self.dilation_rates,
                                          padding=self.padding,
                                          name='conv_{}'.format(i))
            output = ElementwiseLayer('SUM')(output, tensor)
        return output

# Extract slice from tensor to predict
def extract_tensor(input_tensor,size_res):
    input_shape = input_tensor.shape.as_list()
    begin = [0] * len(input_shape)
    begin[1] = size_res
    size = input_shape
    size[1] = size[1] - 2 * size_res
    output_tensor = tf.slice(input_tensor, begin, size)
    return output_tensor
# Layer of convolution after each residual block
class Layer(TrainableLayer):
    def __init__(self,n_output_chans,stride =[1,1,1], kernel_size,padding,acti_func,name):
        self.acti_func = acti_func
        self.name = name
        super(ENet, self).__init__(name=name)
        self.n_output_chans = n_output_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def selu(x, name):
        alpha = 1.6732
        scale = 1.0507
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    
    def layer_op(self, tensor):
        conv_layer = tf.nn.convolution(input=tensor,
                                          filter=self.kernel_size,
                                          strides=self.stride,
                                          padding=self.padding,
                                          name=self.name)
        
        bn_layer = tf.layers.batch_normalization(input=conv_layer,name = 'bn_{}'.self.name)
        output = self.selu(bn_layer,name='acti_{}'.self.name)
        return output
# Layer of prediction   
class Multi_scale(TrainableLayer):
    def __init__(self,n_output_chans,stride =[1,1,1], kernel_size,padding,acti_func,name):
        self.acti_func = acti_func
        self.name = name
        super(ENet, self).__init__(name=name)
        self.n_output_chans = n_output_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    
    def layer_op(self, tensor):
        output = tf.nn.convolution(input=tensor,
                                          filter=self.kernel_size,
                                          strides=self.stride,
                                          padding=self.padding,
                                          name=self.name)
        return output