#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some examples of using the code

Created on Thu Feb 25 15:05:37 2021

@author: mt
"""

#run this cell before any other!

import os
import numpy as np
import matplotlib.pyplot as plt
import Penny as pen
from pygadgetreader import * # pygadgetreader requires a separate import 



#%% Data loading
# loadMost( path, type )
# function loads most of the data contained in gadget snapshot for a particle of given type

Data = pen.loadMost("../Data/snapshot_500", "gas")
for i, key in enumerate(Data.keys()):
    print(i, key)
    print(Data[key])
    print("\n")

# it is often useful to rename commonly used variables for simpler syntax
vgas = Data["vel"]
print(vgas)
    
#%% Select data loading

# loadMost makes use of loader_f which uses readsnap from pygadgetreader
# Using these functions we can access only some desired set of data.

from pygadgetreader import * # pygadgetreader requires a separate import 

pos = readsnap( "../Data/snapshot_500", "pos",  "gas")

Data = pen.loader_f("../Data/snapshot_500", "gas", ["pos", "vel"])
for i, key in enumerate(Data.keys()):
    print(i, key)
    print(Data[key])
    print("\n")

#%% Code Unit access
# You should check and change code units if they do not agree with your model

from pygadgetreader import * # pygadgetreader requires a separate import 

mass = readsnap( "../Data/snapshot_500", "mass",  "gas") * pen.UnitMass_in_g


#%% basic plotting
# Some basic plotting functions are also included
# These require setting up the data in advance

# plt.scatter is somewhat slower than plt.plot
# So if you want to save a little time and resources use plt.plot with dots instead.
# scatterPlot just makes sure you are plotting with dots and no lines

plt.figure(0)
pos = readsnap( "../Data/snapshot_500", "pos",  "gas")
pen.scatterPlot(pos[:,0], pos[:,1])
plt.axis("equal")
plt.xlim(-5,5)
plt.ylim(-5,5)

# plt.scatter has its uses - for example, here each point is coloured based on particle density
plt.figure(1)
rho = readsnap( "../Data/snapshot_500", "rho",  "gas")
plt.scatter(pos[:,0], pos[:,1], c=np.log10(rho))
plt.axis("equal")
plt.xlim(-5,5)
plt.ylim(-5,5)

# But if you want an accurate density map - use plotsnap (ex. density_plotter.py)

#%% RADIAL PLOT
# You can plot a radial profile of some quantity using plotprofile()

pathbase = "../Data/"
data_p = pathbase+'snapshot_500'

snaptime = readhead(data_p,'time') * pen.UnitTime_in_s/ pen.year

Data = pen.loadMost(data_p, "gas")
pos = Data["pos"]
rho = Data["rho"]

quantity = 'density' #here, 'quantity' only determines the output filename, you should input the correct quantity in the plotprofile call
savepath = data_p
fname = savepath + '_' + quantity + "_profile"+".png"

#plt.figure(2) #this shouldn't be required, but if you find that plotprofile overplots on a previous figure, uncomment this line
pen.plotprofile(pos, rho * pen.UnitDensity_in_cgs, snaptime, fname, xlabel="r /pc", ylabel="$\\rho$", xmin=0.001, xmax=5, ymin = 1e-25, ymax = 1e-21, nbins=50, logX=True, logY=True, meanLine=False, medianLine=True,  saveplot=False)

#%% we can add a possibly helpful median or mean line
#plt.figure(3) #this shouldn't be required, but if you find that plotprofile overplots on a previous figure, uncomment this line
pen.plotprofile(pos, rho * pen.UnitDensity_in_cgs,snaptime, fname, xlabel="r /pc", ylabel="$\\rho$", xmin=0.001, xmax=5, nbins=50, logX=True, logY=True, meanLine=True, medianLine=False,  saveplot=False)

#%% there are also some useful manipulation functions (located in basic.py)
# rval is for getting radial values
rgas = pen.rval(pos)
print( "pos:",pos[0,:],"r:",rgas[0])

# getL is mainly used to get angular momenta, but could be used for any a cross product

vel = readsnap( "../Data/snapshot_500", "vel",  "gas")
L = pen.getL(pos, vel) 


# there is also a function for conversion to spherical coordinates
pos_sp = pen.ToSph(pos)


# then  CartRot can be used for rotation of Cartesian coordinates around spherical angles
theta = 0
phi = 0

pos_new = pen.CartRot(pos, theta, phi)

# These should be used with care  - they are not yet fully tested and sometimes give strange results




