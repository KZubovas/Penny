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
#pathbase = "/home/kz/projects/part2_project/test_better/"
#pathbase = "/home/kz/projects/students_turbAGN/test_nt_L1T1R1/"
#pathbase = "../Data/"
#data_p =  [pathbase+"snapshot_040"]#,pathbase+"snapshot_040"]#,pathbase+"snapshot_040",pathbase+"snapshot_045"]#,pathbase+"snapshot_080"]
#data_p = ['/home/kz/projects/students_turbAGN/encons_test_pot_cleanup/snapshot_010']
data_p = ['/home/kz/projects/students_turbAGN/new_L1T1/snapshot_030']
#data_p = ['/home/kz/gadget_mb_2017/test_L1T1/snapshot_030']
#data_p = ['/home/kz/galax_storage/students_turbAGN/new_L2T1/snapshot_037']
#data_p = ['/home/kz/projects/CODE_turbsf_testing/outflow_fg01_nocool/snapshot_030']
#data_p = ['/mnt/old_kz/CODE_turbsf/outflow_fg01_L1T1/snapshot_030']

plane = 'YZ'
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]
extentcube = 0.7
#plt.ioff()
depth = 0.1
#savepath = pathbase
for i, sn in enumerate(data_p):
    if extentcube > 0:
        extent = [-extentcube, extentcube, -extentcube, extentcube]
    else:
        extent = [-5, 5, -5., 5.]
    rho, snaptime = pen.make_Dmap_data(sn, extent, depth, quantity, plane)
    fname = data_p[i]+"_"+quantity+"_slice.png"
#    fname = savepath + quantity + sn[-4:]+"_slice.png"
    
    try:
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname, maxval=1, mamiratio=1e5,saveplot=True)
    except FileNotFoundError:
        os.popen("mkdir ../plot_test/")
        pen.plotsnap(rho*pen.UnitColumnDensity_in_cgs,snaptime,extent,quantity,plane,fname)

#plt.ion()