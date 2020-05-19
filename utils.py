# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:50:43 2020

@author: James
"""

import os;

MAIN_DIRECTORY = os.path.dirname(os.path.dirname(__file__))

def path(path):
    return MAIN_DIRECTORY.join(path)