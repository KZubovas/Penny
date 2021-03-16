"""
Tools for plotting data
"""

import numpy as np
import matplotlib.pyplot as plt
from .basic import *
import scipy.spatial as sc
import Penny.Units as unt

def plotsnap(rho, snaptime, extent, quantity, plane, fname, maxval=999, mamiratio=100, saveplot=False):
    
    if saveplot==False:
        print("saveplot==False:\nNot saving and not closing")

    if maxval==999:        
        ma = rho.max()*0.85# 
    else:
        ma = maxval
    mi = ma/mamiratio# 
    rho_valn = rho.copy()
    rho_valn[rho_valn < mi] = mi
    rho_valn[rho_valn > ma] = ma

        #apibreziu paveiksliuko aplinka
    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        #ax1 = plt.subplots()
        #nubreze zemelapi
    if quantity == 'density':
        colour='Blues'
    if quantity == 'temperature':
        colour='Reds'

    cax1 = ax1.imshow(np.log10(rho_valn),cmap=colour,extent=extent, vmin=np.log10(mi), vmax=np.log10(ma), interpolation = 'Gaussian',origin='lower')
    cbax = fig.colorbar(cax1,fraction=0.046, pad=0.04)
        #dar paveikslelio parametru
#ax1.text(4.5,-6, '$\log{ (\\rm{g cm}^{-2}) }$')
    ax1.scatter(0,0,marker = '+',c='k')
#ax1.ticklabel_format(fontsize = 4)
    if unt.UnitLength_in_cm == 3.086e18:
        if plane=='XY':
            ax1.set_xlabel('$x$ /pc', fontsize=10)
            ax1.set_ylabel('$y$ /pc', fontsize=10)
        if plane=='YZ':
            ax1.set_xlabel('$y$ /pc', fontsize=10)
            ax1.set_ylabel('$z$ /pc', fontsize=10)
        if plane=='XZ':
            ax1.set_xlabel('$x$ /pc', fontsize=10)
            ax1.set_ylabel('$z$ /pc', fontsize=10)
    elif unt.UnitLength_in_cm == 3.086e21:
        if plane=='XY':
            ax1.set_xlabel('$x$ /kpc', fontsize=10)
            ax1.set_ylabel('$y$ /kpc', fontsize=10)
        if plane=='YZ':
            ax1.set_xlabel('$y$ /kpc', fontsize=10)
            ax1.set_ylabel('$z$ /kpc', fontsize=10)
        if plane=='XZ':
            ax1.set_xlabel('$x$ /kpc', fontsize=10)
            ax1.set_ylabel('$z$ /kpc', fontsize=10)
    else:
        if plane=='XY':
            ax1.set_xlabel('$x$', fontsize=10)
            ax1.set_ylabel('$y$', fontsize=10)
        if plane=='YZ':
            ax1.set_xlabel('$y$', fontsize=10)
            ax1.set_ylabel('$z$', fontsize=10)
        if plane=='XZ':
            ax1.set_xlabel('$x$', fontsize=10)
            ax1.set_ylabel('$z$', fontsize=10)
    
    ax1.tick_params(right = 'on',tickdir = 'in',top = 'on')
    ax1.tick_params(axis='both',which = 'major',labelsize = 8)
    xl = abs(extent[1]-extent[0])*0.65+extent[0]
    yl = abs(extent[3]-extent[2])*0.9+extent[2]
    string = str(f'{int(snaptime/1000)*0.001:.2f}') + ' Myr'
    ax1.text(xl,yl,string, fontsize=8, c='k')
    ax1.locator_params(axis='both', nbins = 5)   #which = 'major',
        #Parodo
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    #plt.show()
    


def plotprofile(pos, data, snaptime, fname, xlabel="x", ylabel="y", xmin=0, xmax=100, nbins=50, logTrue=True, meanLine=True, medianLine=False,  saveplot=False):

    if meanLine&medianLine:
        print("meanLine or medianLine should be true at the same time\n ploting mean\n")
    
    if saveplot==False:
        print("saveplot==False:\nNot saving and not closing")

    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    rgas = rval(pos)
    rgas = rgas.reshape(-1, 1)
    
    if meanLine|medianLine:
        Tree = sc.cKDTree(rgas)
        xbins = np.linspace(xmin,xmax,nbins)
        dist = (xbins[1]-xbins[0])/2
        xbins_c = xbins[:-1] + dist
    
        Ind = Tree.query_ball_point( x=xbins_c.reshape(-1,1) , r=dist)
        data_plt = np.zeros(xbins_c.shape[0])
        for i, ind in enumerate(Ind):
            #print(i, ind, rho[ind].mean())
            if medianLine:
                data_plt[i] = np.median(data[ind])
            if meanLine:
                data_plt[i] = data[ind].mean()

    plt.plot(rgas, data, alpha=0.01, ls="", marker=".")
    if meanLine|medianLine:
        ax1.plot(xbins_c, data_plt, c="r")
        #print(xbins_c)
    ax1.set_xlim(xmin,xmax)
    if logTrue:
        ax1.set_yscale("log")
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
        #Parodo
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    #plt.show()