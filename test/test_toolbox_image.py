#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:00:49 2019

@author: itodaiber
"""

import numpy as np
import matplotlib.pyplot as plt

from lib.toolbox import toolbox
from lib.toolbox import images as ti


def test_2d_correlation_and_max_pos():
    
    def show_intersection_and_calc_relative_max_pos(img, mm_per_pixel, img_path = "", axis=0):
        
        size = np.array(np.shape(img))
        middle = (size/2).astype(int)
    
        max_pos, std = ti.get_max_position(img)
        print("max position: {}, {}".format(max_pos[0], max_pos[1]))
        print("std: {}".format(std))
        
        relative_pos = np.subtract(max_pos, middle)
        relative_pos = relative_pos * mm_per_pixel
        
        intersection = ti.get_intersection(img, axis=axis)
        
        maximum = np.max(intersection)
        intersection = intersection / maximum
        
        f = plt.figure()
        plt.plot(intersection)
        plt.ylabel("pixel value / {}".format(maximum))
        plt.xlabel("pixel number / 1")
        plt.title(img_path)
        plt.text(0,0.900, "x-shift: {:.2f} um".format(relative_pos[0]*1000))
        plt.text(0,0.850, "y-shift: {:.2f} um".format(relative_pos[1]*1000))
        plt.text(0,0.800, "x- std: {:.2f} um".format(std[0]*mm_per_pixel*1000))
        plt.text(0,0.750, "y- std: {:.2f} um".format(std[1]*mm_per_pixel*1000))
        plt.show()
        
        if img_path != "":
            f.savefig("{}_intersection.pdf".format(img_path[:-4]))
        
        return relative_pos, std
    
    # generate test data
    rect = np.zeros((100,100))
    rect[44:54,44:54] = 1
    
    rect_2 = np.zeros((100,100))
    
    x_shift = 6
    y_shift = 3
    x_width = 10
    y_width = 15
    
    rect_2[44+x_shift:44+x_width+x_shift,44+y_shift:44+y_width+y_shift] = 1
    
    # expectation
    ex_shift_rel = np.array([6,5.5])
    ex_x_std = np.std(list(range(101,102)))
    ex_y_std = np.std(list(range(101,107)))
    ex_std = np.array([ex_x_std, ex_y_std])
    
    # test functions
    cross = ti.get_cross_correlation_2d(rect_2, rect)
    
    relative_pos, std = show_intersection_and_calc_relative_max_pos(cross, 1, axis=1)
    
    
    toolbox.print_program_section_name("Evaluation")
    counter = 0
    if relative_pos[0] == ex_shift_rel[0]:
        print("x position: passed the test")
    else:
        counter += 1
        print("x position: test failed")
        
    if std[0] == ex_std[0]:
        print("x std: passed the test")
    else:
        counter += 1
        print("x std: test failed")
        
    if relative_pos[1] == ex_shift_rel[1]:
        print("y position: passed the test")
    else:
        counter += 1
        print("y position: test failed")
        
    if std[1] == ex_std[1]:
        print("y std: passed the test")
    else:
        counter += 1
        print("y std: test failed")
    
    if counter == 0:
        print("\nall tests: OK")
    else:
        print("\n{} tests failed".format(counter))
    
test_2d_correlation_and_max_pos()