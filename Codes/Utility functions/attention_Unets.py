"""
Authors: Lorenzo Nava and Kushanav Bhuyan
"""

import cv2 
import time
import os
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import glorot_normal, random_normal, random_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# import segmentation_models as sm

from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.applications import VGG19, densenet
from tensorflow.keras.models import load_model

import numpy as np
import tensorflow as tf 
import losses 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
from sklearn.model_selection import train_test_split

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'glorot_normal'
def accuracy(y_true, y_pred, threshold=0.5):
    """compute accuracy"""
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.equal(K.round(y_true), K.round(y_pred))

# K.round() returns the Element-wise rounding to the closest integer!!!
# So the threshold to determine a true positive is set here!!!!!
# K.sum() returns a single integer output unlike the K.round() which returns an element-wise matrix

def true_positives(y_true, y_pred, threshold=0.5):
    """compute true positive"""
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round(y_true * y_pred)

def false_positives(y_true, y_pred, threshold=0.5):
    """compute false positive"""
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((1 - y_true) * y_pred)

def true_negatives(y_true, y_pred, threshold=0.5):
    """compute true negative"""
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((1 - y_true) * (1 - y_pred))

def false_negatives(y_true, y_pred, threshold=0.5):
    """compute false negative"""
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((y_true) * (1 - y_pred))

def recall_m(y_true, y_pred):
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    recall = K.sum(tp) / (K.sum(tp) + K.sum(fn)+ K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    precision = K.sum(tp) / (K.sum(tp) + K.sum(fp)+ K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Collect all the metrics
metrics = [accuracy, precision_m, recall_m, f1_m]

def unet(opt,input_size, lossfxn):   
  
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu',  kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=opt, loss=lossfxn, metrics=metrics)
    return model

'''
Code credits: https://arxiv.org/abs/1810.07842
	'''

def expend_as(tensor, rep,name):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
    y = multiply([upsample_psi, x], name='q_attn'+name)

    result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn

def UnetConv2D(input, outdim, is_batchnorm, name):
	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
	if is_batchnorm:
		x =BatchNormalization(name=name + '_1_bn')(x)
	x = Activation('relu',name=name + '_1_act')(x)

	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
	if is_batchnorm:
		x = BatchNormalization(name=name + '_2_bn')(x)
	x = Activation('relu', name=name + '_2_act')(x)
	return x


def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name = name + '_act')(x)
    return x

# plain old attention gates in u-net, NO multi-input, NO deep supervision
def attn_unet(opt,input_size, lossfxn):   
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')
    
    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    out = Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)
    
    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=opt, loss=lossfxn, metrics=metrics)
    return model


#regular attention unet with  deep supervision - exactly from paper (my intepretation)
def attn_reg_ds(opt,input_size, lossfxn):
  
    img_input = Input(shape=input_size, name='input_scale1')

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    
    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
 
    loss = {'pred1':lossfxn,
            'pred2':lossfxn,
            'pred3':lossfxn,
            'final': lossfxn}
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[losses.dsc])
    return model


# model proposed in my paper - improved attention u-net with multi-scale input pyramid and deep supervision

def attn_reg(opt,input_size, lossfxn):
    
    img_input = Input(shape=input_size, name='input_scale1')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 64, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    
    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
 
    loss = {'pred1':lossfxn,
            'pred2':lossfxn,
            'pred3':lossfxn,
            'final': losses.tversky_loss}
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[losses.dsc])
    return model
