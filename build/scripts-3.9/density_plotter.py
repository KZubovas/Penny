#!/home/mt/pea/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:31:10 2020

@author: kz
"""

import matplotlib.pyplot as plt
import Penny as pen
import os

data_p = os.popen("ls ../Data/snapshot_*").read().split()

plane = 'XY'
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]
extentcube = 5
#plt.ioff()
depth = 5
savepath = "../plot_test/"
for i, sn in enumerate(data_p):
    if extentcube > 0:
        extent = [-extentcube, extentcube, -extentcube, extentcube]
    else:
        extent = [-5, 5, -5., 5.]
    rho, snaptime = pen.make_Dmap_data(sn, extent, depth, quantity, plane)
    fname = savepath + quantity + sn[-4:]+".png"
    
    try:
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname)
    except FileNotFoundError:
        os.popen("mkdir ../plot_test/")
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname)

#plt.ion()
