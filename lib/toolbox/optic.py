#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:46:40 2019

@author: itodaiber
"""

import math

def calclate_NA(distance, x):
    """
    Calculates the numerical aperture.
    
    Arguments
    ----
        distance: real
        
        x: real
    
    Formular
    -----
    >>>        /           |
    >>>       /            |
    >>>      / theta       x
    >>>     /_____dist_____|
     
    >>> theta = math.atan(x/distance)
    >>> NA = math.sin(theta)
    
    Returns
    ----
        the numerical aperture
        
    """    
    theta = math.atan(x/distance)
    NA = math.sin(theta)
    
    return NA