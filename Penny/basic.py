#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic functions used in various other routines

Created on Tue Feb 23 18:50:45 2021

@author: kz
"""

import numpy as np
import matplotlib.pyplot as plt

def rval(arr):
    return np.sum(arr*arr ,1)**0.5

def getL(pos, vel):
    L = np.zeros_like(pos)
    L[:,0] = pos[:,1] * vel[:,2] - pos[:,2] * vel[:,1]
    L[:,1] = pos[:,2] * vel[:,0] - pos[:,0] * vel[:,2]
    L[:,2] = pos[:,0] * vel[:,1] - pos[:,1] * vel[:,0]
    return L

def dotp(v1, v2): #dot product of two vectors
    return np.sum(v1*v2 ,1)
    
#Tranformation function from Cartesian to spherical coordinates
def ToSph(pos):
    possph = np.zeros_like(pos)
    r = np.sum(pos**2, 1)**0.5
    theta = np.arctan(pos[:,1]/pos[:,0])
    phi = np.arctan(r/pos[:,2])
    possph[:,0] = r
    possph[:,1] = theta
    possph[:,2] = phi
    return possph

#Rotation by theta and phi, takes in and returns Cartesian coordinates
def CartRot(pos, theta, phi):    
    if pos.shape == (3,):
        pos = pos.reshape(1,3)
        
    posn = np.zeros_like(pos)
    #Rot by phi (around z)
    posn[:, 0] = np.cos(phi) * pos[:,0] - np.sin(phi) * pos[:,1]  
    posn[:, 1] = np.sin(phi) * pos[:,0] + np.cos(phi) * pos[:,1]  
    posn[:, 2] = pos[:,2]
    pos = 0
    pos = posn.copy()
    
    #Rot by theta (around y)
    posn[:, 0] = np.cos(theta) * pos[:,0] + np.sin(theta) * pos[:,2]  
    posn[:, 1] = pos[:,1]
    posn[:, 2] = np.cos(theta) * pos[:,2] - np.sin(theta) * pos[:,0]
    return posn

def scatterPlot(x, y, alpha=0.1, marker="."):
    if len(x) == 1:
        print("arrays must have sizes >1; if you need to plot a single point, use plt.scatter")
    else:
        plt.plot(x, y, ls="", marker=marker, alpha=alpha)