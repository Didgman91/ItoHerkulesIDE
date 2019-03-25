#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:26:33 2019

@author: itodaiber
"""

import numpy as np

def get_polynomial_fit(x, y, deg):
    """
    Arguments
    ----
        x : array_like, shape (M,)
            x-coordinates of the M sample points (x[i], y[i]).
        
        y : array_like, shape (M,) or (M, K)    
            y-coordinates of the sample points. Several data sets of sample points sharing the same x-coordinates can be fitted at once by passing in a 2D-array that contains one dataset per column.
        
        deg : int    
            Degree of the fitting polynomial
            
    Returns
    ----
        p: numpy.poly1d
            A one-dimensional polynomial class.

    """
    
    z = np.polyfit(x,y,deg)
    
    p = np.poly1d(z)
    
    return p