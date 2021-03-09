#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:31:10 2020

@author: kz
"""

import matplotlib.pyplot as plt
import Penny as pen
import os

#data_p = os.popen("ls ../Data/snapshot_*").read().split()

#data_p = os.popen("ls /home/kz/projects/part2_project/test1/snapshot_020").read().split()
#data_p = ["/home/kz/projects/part2_project/test1/snapshot_020","/home/kz/projects/part2_project/test1/snapshot_030","/home/kz/projects/part2_project/test1/snapshot_040"]

pathbase = "/home/kz/projects/part2_project/test2/"
data_p = [pathbase+"snapshot_020",pathbase+"snapshot_030",pathbase+"snapshot_040",pathbase+"snapshot_060",pathbase+"snapshot_080"]

plane = 'XY'
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]
extentcube = 1
#plt.ioff()
depth = 0.5
savepath = pathbase
for i, sn in enumerate(data_p):
    if extentcube > 0:
        extent = [-extentcube, extentcube, -extentcube, extentcube]
    else:
        extent = [-5, 5, -5., 5.]
    rho, snaptime = pen.make_Dmap_data(sn, extent, depth, quantity, plane)
    fname = savepath + quantity + sn[-4:]+".png"
    
    try:
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname,saveplot=True)
    except FileNotFoundError:
        os.popen("mkdir ../plot_test/")
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname)

#plt.ion()
