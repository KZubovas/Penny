#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:31:10 2020

@author: kz
"""

import gadget_helper as gh
#import os 
#import numpy as np
import matplotlib.pyplot as plt

  




#path_start = "/home/kz/projects/sgra_feedback/test_betterbeta_novisclim/"
path_start = "/home/kz/projects/students_turbAGN/test_L1T1_rot/"
snapmin = 40
snapmax = 41
deltasnap = 2

plane = 'XZ'
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]

extentcube = 1
depth = 40

plt.ioff() #use this to prevent each snapshot being shown on screen

for snapnum in range(snapmin, snapmax, deltasnap):

    path = path_start+'snapshot_%03d'%snapnum
    if extentcube > 0:
        extent = [-extentcube, extentcube, -extentcube, extentcube]
    else:
        extent = [-20, 20, -20., 20.]

    rho, snaptime = gh.make_Dmap_data(path, extent, depth, quantity, plane)

    fname = path_start+quantity+'%03d'%snapnum+'_'+plane+'.png'

    gh.plotsnap(rho,snaptime,extent,quantity,plane,fname)

plt.ion() #use this to change back to showing things on screen when plots are made