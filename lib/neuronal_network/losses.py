#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:42:09 2018

@author: daiberma
"""

import functools
from keras import backend as K
import tensorflow as tf

import numpy as np


#from keras_contrib.losses import jaccard_distance


def as_keras_metric(method):
    """ wrapper function for tensorflow functions to call these in keras
    
    # Arguments
        method: wrapped function
    """
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        K.get_session().run([value, update_op])
        K.get_session().run([value])
#        K.get_session().run(tf.local_variables_initializer())
#        with tf.control_dependencies([update_op]):
#            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def pearson_correlation_coefficient(y_true, y_pred):
    """calculates the Pearson correlation coefficient
    # Arguments
        y_true
            ground truth data
        y_pred
            prediction
    
    # Returns
        the Pearson correlation coefficient
    """
    
    if type(y_true) is type(np.array([])):
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, tf.float32)
    
    if type(y_pred) is type(np.array([])):    
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.cast(y_pred, tf.float32)
    
    return tf.contrib.metrics.streaming_pearson_correlation(y_true, y_pred)



def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# Aliases.
jd = JD = jaccard_distance
pcc = PCC = pearson_correlation_coefficient