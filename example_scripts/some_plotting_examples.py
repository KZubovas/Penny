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

#pathbase = "../Data/"
#data_p = pathbase+'snapshot_500'
#data_p = '/home/kz/projects/mato_code/test2/snapshot_010'
#data_p = '/home/kz/projects/gadget_archyvinis/test7/snapshot_001'
#data_p = '/home/kz/projects/students_turbAGN/dm_L3T1/snapshot_060'
#data_p = '/home/kz/projects/students_turbAGN/new_L1T1/snapshot_020'
#data_p = '/home/kz/projects/CODE_turbsf_testing/outflow_fg01_newtest/snapshot_020'
#data_p = '/mnt/old_kz/CODE_turbsf/test_L1T1_copy/snapshot_001'
#data_p = '/mnt/old_kz/CODE_turbsf/outflow_fg01_L1T1/snapshot_020'
#data_p = '/mnt/old_kz/gadget-mb-latest/test_L1T1_copy/snapshot_001'
#data_p = '/home/kz/gadget_mb_2017/test_L1T1/snapshot_040'
#data_p = '/home/kz/gadget_mb_2017/mato_test/snapshot_000'
#data_p = '/home/kz/projects/turbsf/test_L1T1_copy/snapshot_010'
#data_p = '/home/kz/matas_storage/fermi_cubic_fg1e-3_highN.dat'
data_p = '/home/kz/projects/students_turbAGN/new_L1T1_correct_cooling/snapshot_030'
#data_p = '/home/kz/projects/gadget_sn_2021/test_mato/snapshot_001'

#NPart = readhead(data_p, 'npartTotal')
snaptime = readhead(data_p,'time') * pen.UnitTime_in_s/ pen.year

Data = pen.loadMost(data_p, "gas")
pos = Data["pos"]
rtot = Data["rtot"]
rho = Data["rho"]
vr = Data["vrad"]
mass = Data["mass"]
vtot = Data["vtot"]
u = Data["u"]
temp = u*pen.u_to_temp_fac

tempc = (np.log10(temp)-min(np.log10(temp)))/(max(np.log10(temp))- min(np.log10(temp)))

# plt.plot(pos[:,0], pos[:,1], ls='', marker='.', markersize=0.1, c=tempc)
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.yscale("linear")
# plt.show()

plt.plot(rtot, temp, ls='', marker='.', markersize=0.1)
plt.xlim(0,2)
plt.ylim(1e1,1e7)
plt.yscale("log")
plt.show()

# plt.plot(rtot, rho, ls='', marker='.', markersize=0.1)
# plt.xlim(0,20)
# plt.ylim(1e-6,3e-3)
# plt.yscale("log")
# plt.show()

# plt.plot(rho, temp, ls='', marker='.', markersize=0.1)
# #plt.xlim(1e-5,1e-3)
# #plt.ylim(1e2,1e5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Density")
# plt.ylabel("Temperature")
# plt.show()

# plt.plot(rtot, vr*pen.UnitVelocity_in_cm_per_s/1.e5, ls='', marker='.', markersize=0.1)
# plt.xlim(0,20)
# plt.ylim(-400,200)
# plt.yscale("linear")
# plt.show()
 
#%%

data_p = '/home/kz/projects/students_turbAGN/dm_control/snapshot_096'
snaptime = readhead(data_p,'time') * pen.UnitTime_in_s/ pen.year

Data_s = pen.loadMost(data_p, "dm")
pos_dm = Data_s["pos"]
rtot_dm = Data_s["rtot"]
mass_dm = Data_s["mass"]
vtot_dm = Data_s["vtot"]

#enbig = 3.42026e58 

#eninput = 1.255e46*3.15e13*0.05 / pen.UnitEnergy_in_cgs
#eninit = (mass*vtot**2/2).sum()+(mass*np.log10(rtot)).sum()+(mass*u).sum()
# enfinal_s = (mass_s*vtot_s**2/2).sum()+(mass_s*np.log10(rtot_s)).sum()

#mask = (vr > 1) #& (temp < 5e4)
#posmask = pos[mask]
#mdot = mass[mask]*vr[mask]/rtot[mask] * pen.UnitMass_in_g / 1.989e33 / pen.UnitTime_in_s * 3.15e7
#pdot = mdot * 1.989e33 / 3.15e7 * vr[mask] * pen.UnitVelocity_in_cm_per_s / (1.3e46 / 3e10)

quantity = 'mdot' #here, 'quantity' only determines the output filename, you should input the correct quantity in the plotprofile call
savepath = data_p
fname = savepath + '_' + quantity + "_profile"+".png"

#plt.figure(2) #this shouldn't be required, but if you find that plotprofile overplots on a previous figure, uncomment this line
pen.plotprofile(pos_dm, mass_dm, snaptime, fname, xlabel="r /kpc", ylabel="Mdm", xmin=0.01, xmax=5, ymin = 0, ymax = 0, nbins=50, logX=True, logY=True, meanLine=True, medianLine=False,  saveplot=True)

#this plots "shell sums" of a quantity rather than simple radial profiles; useful for such quantities as mass flow rate
#pen.plotsum(pos_dm, mass_dm/rtot_dm**2, snaptime, fname, xlabel="r /kpc", ylabel="$\\dot{M}$ / $M_{Sun} yr^{-1}$", xmin=0.01, xmax=5, ymin = 0.1, ymax = 100, nbins=20, SumByShell=True, logX=True, logY=True, deviations=True, saveplot=True)

#ylabel="$\\dot{p}$ / $L_{Edd} c^{-1}$"


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




