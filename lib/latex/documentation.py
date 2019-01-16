#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:13:28 2018

@author: maxdh
"""

from lib.toolbox import toolbox


def generate_documentation():
    toolbox.run_process("pdflatex",
                        ["-synctex=1",
                         "-interaction=nonstopmode",
                         "\"main\".tex"],
                         working_dir="/lib/latex/document")