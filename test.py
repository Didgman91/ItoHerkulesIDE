#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:19:12 2018

@author: daiberma
"""

import os

import numpy as np

from scipy import signal

from lib.toolbox import toolbox
from lib.toolbox import images as ti

from script import memory_effect as me

import matplotlib.pyplot as plt


from lib.latex import documentation


from lib.module_base import module_base

from lib.toolbox import math as tm


import cv2


x = np.array([1,2,3,4,5])
y = np.array([-2.1,-2.9,-4.0,-5.1,-6])

plot_settings = {'suptitle': 'shift',
                  'xlabel': 'distance / m',
                  'xmul': 1,
                  'ylabel': 'calculated shift / um',
                  'ymul': 1,
                  'delimiter': ',',
                  'skip_rows': 1}

def fit(path, plot_settings, x_column = 0, y_column = [1],
        fit_section=[0,0]):
    
    a = np.loadtxt(path, delimiter=plot_settings['delimiter'],
                       skiprows=plot_settings['skip_rows'])

    p = []
    for y in y_column:
        label = toolbox.get_file_name(path) + "_c{}".format(y)
        x_value = a[:,x_column] * plot_settings['xmul']
        y_value = a[:,y] * plot_settings['ymul']
        
        plt.plot(x_value, y_value, '.', label=label)

        if fit_section == [0,0]:
            p1 = tm.get_polynomial_fit(x_value, y_value,1)
        else:
            start = np.where(x_value >= fit_section[0])[0]
            if fit_section[1] > fit_section[0]:
                stop = np.where(x_value <= fit_section[1])[0]
                r = toolbox.get_intersection(start.tolist(), stop.tolist())
                start = r[0]
                stop = r[-1]
            else:
                start = start[0]
                stop = len(x_value)-1
            p1 = tm.get_polynomial_fit(x_value[start:stop],
                                       y_value[start:stop],1)
            
            plt.axvspan(x_value[start], x_value[stop], facecolor='gray',
                        alpha=0.15)
        
        print("{} [{}, {}]".format(label, fit_section[0], fit_section[1]))
        print(p1)
        xp = np.linspace(min(x_value), max(x_value))
        plt.plot(xp, p1(xp), '--', label=label + " (fit)")
        
        p += p1
    plt.xlabel(plot_settings['xlabel'])
    plt.ylabel(plot_settings['ylabel'])
    
    plt.legend(loc='best')
    
    plt.suptitle(plot_settings['suptitle'])
    
    return p

path = "data/memory_effect/output/shift/shift_125,0.csv"


p = fit(path, plot_settings, y_column=[2], fit_section=[0.256,0])

