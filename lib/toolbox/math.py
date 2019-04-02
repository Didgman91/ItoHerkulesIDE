#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:26:33 2019

@author: itodaiber
"""

import numpy as np
import matplotlib.pyplot as plt

import lib.toolbox.toolbox as toolbox


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

def csv_fit_and_plot(path, plot_settings, x_column = 0, y_column = [1],
        fit_section=[0,0], plot_save_path=""):
    """
    Arguments
    ----
        path: list<string>
            csv file path list
        plot_settings: dictionary
            refere section Plot settings
        x_column: integer
            specifies a csv column as x-axis
        y_column: list<integer>
            specifies csv columns as y-axes
        fit_section: list<integer>
            specifies a start and a stop value on the x-axis
        plot_save_path: string (optinal)
            saves the plot as a pdf with this path
        
    Plot settings
    -----
        Example plot settings:
            >>> plot_settings = {'suptitle': 'shift',
            >>>                  'xlabel': 'distance / m',
            >>>                  'xmul': 1,
            >>>                  'ylabel': 'calculated shift / um',
            >>>                  'ymul': 1000,
            >>>                  'delimiter': ',',
            >>>                  'skip_rows': 1}
    
    Returns
    ----
        p: list<numpy.poly1d>
            A list of one-dimensional polynomial class.
    """
    
    p = plt.figure()
    poly = []
    for f in path:
        a = np.loadtxt(f, delimiter=plot_settings['delimiter'],
                           skiprows=plot_settings['skip_rows'])

        
        for y in y_column:
            label = toolbox.get_file_name(f) + "_c{}".format(y)
            x_value = a[:,x_column] * plot_settings['xmul']
            y_value = a[:,y] * plot_settings['ymul']
            
            line, = plt.plot(x_value, y_value, '.', label=label)

            if fit_section == [0,0]:
                p1 = get_polynomial_fit(x_value, y_value,1)
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
                p1 = get_polynomial_fit(x_value[start:stop],
                                           y_value[start:stop],1)
                
                plt.axvspan(x_value[start], x_value[stop], facecolor='gray',
                            alpha=0.15/len(path))
            
            print("{} [{}, {}]".format(label, fit_section[0], fit_section[1]))
            print(p1)
            xp = np.linspace(min(x_value), max(x_value))
            
            if len(path) == 1:
                plt.plot(xp, p1(xp), '--',
                         label=label + " (fit)\n{:.3}x + {:.3}".format(p1.coefficients[0],
                                         p1.coefficients[1]))
            else:
                plt.plot(xp, p1(xp), '--',
                         color = line.get_color(),
                         label=label + " (fit)\n{:.3}x + {:.3}".format(p1.coefficients[0],
                                         p1.coefficients[1]))
            
            poly += p1
    plt.xlabel(plot_settings['xlabel'])
    plt.ylabel(plot_settings['ylabel'])
    
    if len(path) == 1:
        plt.legend(loc='best')
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    
    plt.suptitle(plot_settings['suptitle'])
    
    if plot_save_path != "":
        p.savefig(plot_save_path, bbox_inches='tight')
    
    plt.close(p)
    return poly

