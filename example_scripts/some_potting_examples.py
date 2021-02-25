#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some examples of using the code

Created on Thu Feb 25 15:05:37 2021

@author: mt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Penny as pen




#%% Data loading
# loadMost( path, type )
# function loads most of the data contained in gadget snapshot for a particle of given type

Data = pen.loadMost("../Data/snapshot_500", "gas")
for i, key in enumerate(Data.keys()):
    print(i, key)
    print(Data[key])
    print("\n")

# it is often useful to rename often used variable for simpler syntax
vgas = Data["vel"]
print(vgas)
    
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

mass = readsnap( "../Data/snapshot_500", "mass",  "gas") * pen.UnitMass_in_g


#%% basic plotting
# Some basic plotting functions are also included
# These require to set up the data in advance

# plt.scatter is somewhat slower than plt.plot
# So if you want to save a little time and resources us, plt.plot with dots instead.
# scatterPlot just mokes sure you are plotting with dots and no lines

plt.figure(0)
pos = readsnap( "../Data/snapshot_500", "pos",  "gas")
pen.scatterPlot(pos[:,0], pos[:,1])
plt.axis("equal")
plt.xlim(-5,5)
plt.ylim(-5,5)

# plt.scatter has its uses
plt.figure(1)
rho = readsnap( "../Data/snapshot_500", "rho",  "gas")
plt.scatter(pos[:,0], pos[:,1], c=np.log10(rho))
plt.axis("equal")
plt.xlim(-5,5)
plt.ylim(-5,5)

# But if you want an accurate density map - use plotsnap (ex. density_plotter.py)

#%%
# You can plot a radial profile of some quantity using plotprofile()

snaptime = readhead("../Data/snapshot_500",'time') * pen.UnitTime_in_s/ pen.year
quantity = 'density' #type of quantity to make a map of, can be 'density', 'temperature' [others tba]
savepath = "../plot_test/"
fname = savepath + quantity + "_profile_500"+".png"

plt.figure(2)
pen.plotprofile(pos, rho * pen.UnitDensity_in_cgs,snaptime, fname, xlabel="r /pc", ylabel="$\\rho$", xmin=0, xmax=10, nbins=50, logTrue=True, meanLine=False, medianLine=False,  saveplot=False)

#%% we can add a possibly helpfull median or mean line
plt.figure(3)
pen.plotprofile(pos, rho * pen.UnitDensity_in_cgs,snaptime, fname, xlabel="r /pc", ylabel="$\\rho$", xmin=0, xmax=10, nbins=50, logTrue=True, meanLine=True, medianLine=False,  saveplot=False)

#%% there are also some usefull manipulation function in (located in basic.py)
# rval is for getting radial values
rgas = pen.rval(pos)
print( "pos:",pos[0,:],"r:",rgas[0])

# getL is mainly used to get angular momenta, but could be used a cross product

vel = readsnap( "../Data/snapshot_500", "vel",  "gas")
L = pen.getL(pos, vel) 


# there is also a function for conversion to spherical coordinates
pos_sp = pen.ToSph(pos)


# then  CartRot can be used for rotation in spherical coordinates
theta = 0
phi = 0

pos_new = pen.CartRot(pos_sp, theta, phi)

# These should be used with care  - they are not yet fully tested and sometimes give strange results




