"""
python implementation of paper: Deep speckle correlation: a deep learning approach towards scalable imaging through
scattering media

paper link: https://arxiv.org/abs/1806.04139

Author: Yunzhe li, Yujia Xue, Lei Tian

Computational Imaging System Lab, @ ECE, Boston University

Date: 2018.08.21
"""

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import numpy as np



## define conv_factory: batch normalization + ReLU + Conv2D + Dropout (optional)
#def conv_factory(x, concat_axis, nb_filter,
#                 dropout_rate=None, weight_decay=1E-4):
#    x = BatchNormalization(axis=concat_axis,
#                           gamma_regularizer=l2(weight_decay),
#                           beta_regularizer=l2(weight_decay))(x)
#    x = Activation('relu')(x)
#    x = Conv2D(nb_filter, (5, 5), dilation_rate=(2, 2),
#               kernel_initializer="he_uniform",
#               padding="same",
#               kernel_regularizer=l2(weight_decay))(x)
#    if dropout_rate:
#        x = Dropout(dropout_rate)(x)
#
#    return x
#
#
## define dense block
#def denseblock(x, concat_axis, nb_layers, growth_rate,
#               dropout_rate=None, weight_decay=1E-4):
#    list_feat = [x]
#    for i in range(nb_layers):
#        x = conv_factory(x, concat_axis, growth_rate,
#                         dropout_rate, weight_decay)
#        list_feat.append(x)
#        x = Concatenate(axis=concat_axis)(list_feat)
#
#    return x
#
#
## define model U-net modified with dense block
#def get_model_deep_speckle():
#    inputs = Input((256, 256, 1))
#    print("inputs shape:", inputs.shape)
#
#    # E0
#    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#    print("conv1 shape:", conv1.shape)
#    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db1 shape:", db1.shape)
#    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
#    print("pool1 shape:", pool1.shape)
#
#    # E1
#    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#    print("conv2 shape:", conv2.shape)
#    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db2 shape:", db2.shape)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
#    print("pool2 shape:", pool2.shape)
#
#    # E2
#    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#    print("conv3 shape:", conv3.shape)
#    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db3 shape:", db3.shape)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
#    print("pool3 shape:", pool3.shape)
#
#    # E3
#    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#    print("conv4 shape:", conv4.shape)
#    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db4 shape:", db4.shape)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)
#    print("pool4 shape:", pool4.shape)
#
#    # --
#    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#    print("conv5 shape:", conv5.shape)
#    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db5 shape:", db5.shape)
#    up5 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db5))
#    print("up5 shape:", up5.shape)
#    merge5 = Concatenate(axis=3)([db4, up5])
#    print("merge5 shape:", merge5.shape)
#
#    # D1
#    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
#    print("conv6 shape:", conv6.shape)
#    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db5 shape:", db6.shape)
#    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db6))
#    print("up6 shape:", up6.shape)
#    merge6 = Concatenate(axis=3)([db3, up6])
#    print("merge6 shape:", merge6.shape)
#
#    # D2
#    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#    print("conv7 shape:", conv7.shape)
#    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db7 shape:", db7.shape)
#    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db7))
#    print("up7 shape:", up7.shape)
#    merge7 = Concatenate(axis=3)([db2, up7])
#    print("merge7 shape:", merge7.shape)
#
#    # D3
#    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#    print("conv8 shape:", conv8.shape)
#    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db8 shape:", db8.shape)
#    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db8))
#    print("up8 shape:", up8.shape)
#    merge8 = Concatenate(axis=3)([db1, up8])
#    print("merge8 shape:", merge8.shape)
#    
#    # D4
#    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#    print("conv9 shape:", conv9.shape)
#    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db9 shape:", db9.shape)
#    
#    # D5
#    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9)
#    print("conv10 shape:", conv10.shape)
#    conv11 = Conv2D(2, 1, activation='softmax')(conv10)
#    print("conv11 shape:", conv11.shape)
#
#    model = Model(inputs=inputs, outputs=conv11)
#
#    return model

# define model U-net modified with dense block
def get_model_deep_speckle_64x64():
    inputs = Input((64, 64, 1))
    print("inputs shape:", inputs.shape)

    # E0
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print("conv1 shape:", conv1.shape)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    # E1
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

#    # E2
#    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#    print("conv3 shape:", conv3.shape)
#    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db3 shape:", db3.shape)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
#    print("pool3 shape:", pool3.shape)
#
#    # E3
#    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#    print("conv4 shape:", conv4.shape)
#    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
#    print("db4 shape:", db4.shape)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)
#    print("pool4 shape:", pool4.shape)

    # --
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    print("conv5 shape:", conv5.shape)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=0.5)
    print("db5 shape:", db5.shape)
    up5 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db5))
    print("up5 shape:", up5.shape)
    merge5 = Concatenate(axis=3)([db2, up5])
    print("merge5 shape:", merge5.shape)

#    # D1
#    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
#    print("conv6 shape:", conv6.shape)
#    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db5 shape:", db6.shape)
#    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db6))
#    print("up6 shape:", up6.shape)
#    merge6 = Concatenate(axis=3)([db3, up6])
#    print("merge6 shape:", merge6.shape)
#
#    # D2
#    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#    print("conv7 shape:", conv7.shape)
#    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
#    print("db7 shape:", db7.shape)
#    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#        UpSampling2D(size=(2, 2))(db7))
#    print("up7 shape:", up7.shape)
#    merge7 = Concatenate(axis=3)([db2, up7])
#    print("merge7 shape:", merge7.shape)

    # D3
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    print("conv8 shape:", conv8.shape)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    print("db8 shape:", db8.shape)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db8))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([db1, up8])
    print("merge8 shape:", merge8.shape)
    
    # D4
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    print("conv9 shape:", conv9.shape)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5)
    print("db9 shape:", db9.shape)
    
    # D5
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9)
    print("conv10 shape:", conv10.shape)
    conv11 = Conv2D(2, 1, activation='softmax', name='predictions')(conv10)
    print("conv11 shape:", conv11.shape)

    model = Model(inputs=inputs, outputs=conv11)
    
#    model = print"Model loaded"
    
    return model


def getLossFunction(y_true, y_pred):
    pixelNumberX = len(y_true[:,0])
    pixelNumberY = len(y_true[0,:])
    N = pixelNumberX * pixelNumberY
    c = len(y_pred[2])
    
    loss = calcLossFunction(N, c, pixelNumberX, pixelNumberY, y_true, y_pred)
    
    return loss

def calcLossFunction(N, channels, pixelNumberX, pixelNumberY, g, p):
    "N: number of pixels; channels: number of channels; g: ground-truth pixel value; p: perdicted pixel value"
    buffer = 0
    
    for x in range(pixelNumberX):
        for y in range(pixelNumberY):
            for c in range(channels):
                buffer += -(g[x, y] * np.log(p[x,y,c]) + (1-g[x,y]) * np.log(1-p[x,y,c]))
    
    buffer /= 2*N
    
    return buffer

def getMetrics(y_true, y_pred):
    
    # calculate the Pearcon product-moment correlation coefficient
    buffer = np.corrcoef(y_true, y_pred[:,:,0]) # c=0: like in the demo.py 
    
    
    return