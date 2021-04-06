#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:31:10 2020

@author: kz
"""

import matplotlib.pyplot as plt
import Penny as pen
import os

#one option of defining paths, based on the contents of the folder
#pathbase = "../Data/"
#data_p = os.popen("ls ../Data/snapshot_*").read().split()

#another, more direct option
pathbase = "../Data/"
data_p =  [pathbase+"snapshot_040"]#,pathbase+"snapshot_040"]#,pathbase+"snapshot_040",pathbase+"snapshot_045"]#,pathbase+"snapshot_080"]

plane = 'XY'
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]
extentcube = 2.
#plt.ioff()
depth = 0.1
savepath = pathbase
for i, sn in enumerate(data_p):
    if extentcube > 0:
        extent = [-extentcube, extentcube, -extentcube, extentcube]
    else:
        extent = [-5, 5, -5., 5.]
    rho, snaptime = pen.make_Dmap_data(sn, extent, depth, quantity, plane)
    fname = savepath + quantity + sn[-4:]+"_slice.png"
    
    try:
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname, maxval=10, mamiratio=1e5,saveplot=True)
    except FileNotFoundError:
        os.popen("mkdir ../plot_test/")
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname)

#plt.ion()
