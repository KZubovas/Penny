"""
Tools for plotting data
"""

import numpy as np
import random
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
    


def plotprofile(pos, data, snaptime, fname, xlabel="x", ylabel="y", xmin=0, xmax=100, ymin=0, ymax=0, nbins=50, logX=True, logY=True, meanLine=True, medianLine=False,  saveplot=False):

    if meanLine&medianLine:
        print("Both meanLine and medianLine should not be true at the same time\n plotting mean line\n")
    
    if saveplot==False:
        print("saveplot==False:\nNot saving and not closing")
        
    if ymin==0:
        ymin = 0.9*min(data)
        
    if ymax==0:
        ymax = 1.1*max(data)

    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    rgas = rval(pos)
    rgas = rgas.reshape(-1, 1)
    
    if meanLine|medianLine:
        if logX:
            Tree = sc.cKDTree(np.log10(rgas))
            xbins = np.linspace(np.log10(xmin),np.log10(xmax),nbins)
        else:
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
        if logX:
            ax1.plot(10**xbins_c, data_plt, c="r")
        else:
            ax1.plot(xbins_c, data_plt, c="r")
        #print(xbins_c)
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(ymin,ymax)
    if logX:
        ax1.set_xscale("log")
    
    if logY:
        ax1.set_yscale("log")
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if logX:
        xl = xmin * (xmax / xmin)**0.1
    else:
        xl = xmin + (xmax - xmin) * 0.1
    if logY:
        yl = ymin * (ymax / ymin)**0.9
    else:
        yl = ymin + (ymax - ymin) * 0.9
    string = str(f'{int(snaptime/1000)*0.001:.2f}') + ' Myr'
    ax1.text(xl,yl,string, fontsize=8, c='k')

        #Parodo
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    #plt.show()



def plotsum(pos, data, snaptime, fname, xlabel="x", ylabel="y", xmin=0, xmax=100, ymin=0, ymax=0, nbins=50, SumByShell=True, logX=True, logY=True, deviations=True, saveplot=False):

    if saveplot==False:
        print("saveplot==False:\nNot saving and not closing")
        

    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    rg = rval(pos)
    rgas = rg.reshape(-1, 1)
    
    if logX:
        Tree = sc.cKDTree(np.log10(rgas))
        xbins = np.linspace(np.log10(xmin),np.log10(xmax),nbins)
    else:
        Tree = sc.cKDTree(rgas)
        xbins = np.linspace(xmin,xmax,nbins)

    dist = (xbins[1]-xbins[0])/2
    xbins_c = xbins[:-1] + dist
    
    Ind = Tree.query_ball_point( x=xbins_c.reshape(-1,1) , r=dist)
    sumline = np.zeros(xbins_c.shape[0])
    if deviations:
        devline = np.zeros(xbins_c.shape[0])
    for i, ind in enumerate(Ind):
        if SumByShell:
            data[ind] *= rg[ind] / (2. * dist)
        #print(i, ind, rho[ind].mean())
        sumline[i] = data[ind].sum()
        if deviations and data[ind].size != 0:
            #create 100 random subsamples of data[ind]
            dummy = np.zeros(100)
            for j in range(100):
                dum_list = np.asarray(random.choices(data[ind], k=int(np.ceil(data[ind].size/100))))
                dummy[j] = dum_list.sum()*data[ind].size/dum_list.size                        
            #calculate deviation of those sums
            devline[i] = np.std(dummy)           

    if ymin==0:
        ymin = 0.9*min(sumline-devline)
    if ymax==0:
        ymax = 1.1*max(sumline+devline)

    if logX:
        plotbins = 10**xbins_c
    else:
        plotbins = xbins_c

    plt.plot(plotbins, sumline, linewidth = 2, c='r')
    if deviations:
        ax1.fill_between(plotbins, sumline-devline, sumline+devline, facecolor='red', alpha=0.3)
        ax1.plot(plotbins, sumline-devline, linestyle='dashdot', linewidth=1, c='r', alpha = 0.8)
        ax1.plot(plotbins, sumline+devline, linestyle='dashdot', linewidth=1, c='r', alpha = 0.8)
            
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(ymin,ymax)
    if logX:
        ax1.set_xscale("log")
    
    if logY:
        ax1.set_yscale("log")
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if logX:
        xl = xmin * (xmax / xmin)**0.1
    else:
        xl = xmin + (xmax - xmin) * 0.1
    if logY:
        yl = ymin * (ymax / ymin)**0.9
    else:
        yl = ymin + (ymax - ymin) * 0.9
    string = str(f'{int(snaptime/1000)*0.001:.2f}') + ' Myr'
    ax1.text(xl,yl,string, fontsize=8, c='k')

        #Parodo
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    #plt.show()



def plotrel(pos, data, posrel, datarel, snaptime, fname, xlabel="x", ylabel="y", xmin=0, xmax=100, ymin=0, ymax=0, nbins=50, reltype="difference", SumByShell=True, logX=True, logY=True, deviations=False, saveplot=False):

    if saveplot==False:
        print("saveplot==False:\nNot saving and not closing")
        

    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    rg = rval(pos)
    rgas = rg.reshape(-1, 1)
    
    rgrel = rval(posrel)
    rgasrel = rgrel.reshape(-1, 1)
    
    if logX:
        Tree = sc.cKDTree(np.log10(rgas))
        Treerel = sc.cKDTree(np.log10(rgasrel))
        xbins = np.linspace(np.log10(xmin),np.log10(xmax),nbins)
    else:
        Tree = sc.cKDTree(rgas)
        Treerel = sc.cKDTree(rgasrel)
        xbins = np.linspace(xmin,xmax,nbins)

    dist = (xbins[1]-xbins[0])/2
    xbins_c = xbins[:-1] + dist
    
    Ind = Tree.query_ball_point( x=xbins_c.reshape(-1,1) , r=dist)
    Indrel = Treerel.query_ball_point( x=xbins_c.reshape(-1,1) , r=dist)
    sumline = np.zeros(xbins_c.shape[0])
    sumlinerel = np.zeros(xbins_c.shape[0])
    if deviations:
        devline = np.zeros(xbins_c.shape[0])
    for i, ind in enumerate(Ind):
        if SumByShell:
            data[ind] *= rg[ind] / (2. * dist)
        #print(i, ind, rho[ind].mean())
        sumline[i] = data[ind].sum()
        if deviations and data[ind].size != 0:
            #create 100 random subsamples of data[ind]
            dummy = np.zeros(100)
            for j in range(100):
                dum_list = np.asarray(random.choices(data[ind], k=int(np.ceil(data[ind].size/100))))
                dummy[j] = dum_list.sum()*data[ind].size/dum_list.size                        
            #calculate deviation of those sums
            devline[i] = np.std(dummy)           
    for i, indrel in enumerate(Indrel):
        if SumByShell:
            datarel[indrel] *= rg[indrel] / (2. * dist)
        #print(i, ind, rho[ind].mean())
        sumlinerel[i] = datarel[indrel].sum()
        
    #now determine what we're actually plotting:
        if reltype=="ratio":
            plotline = sumline/sumlinerel
        elif reltype=="difference":
            plotline = sumline/sumlinerel - 1
        else:
            print(f'I cannot understand this relation type.')
            return None

    if ymin==0:
        ymin = 0.9*min(plotline)
    if ymax==0:
        ymax = 1.1*max(plotline)

    if logX:
        plotbins = 10**xbins_c
    else:
        plotbins = xbins_c

    plt.plot(plotbins, plotline, linewidth = 2, c='r')
    if deviations:
        ax1.fill_between(plotbins, sumline-devline, sumline+devline, facecolor='red', alpha=0.3)
        ax1.plot(plotbins, sumline-devline, linestyle='dashdot', linewidth=1, c='r', alpha = 0.8)
        ax1.plot(plotbins, sumline+devline, linestyle='dashdot', linewidth=1, c='r', alpha = 0.8)
            
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(ymin,ymax)
    if logX:
        ax1.set_xscale("log")
    
    if logY:
        ax1.set_yscale("log")
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if logX:
        xl = xmin * (xmax / xmin)**0.1
    else:
        xl = xmin + (xmax - xmin) * 0.1
    if logY:
        yl = ymin * (ymax / ymin)**0.9
    else:
        yl = ymin + (ymax - ymin) * 0.9
    string = str(f'{int(snaptime/1000)*0.001:.2f}') + ' Myr'
    ax1.text(xl,yl,string, fontsize=8, c='k')

        #Parodo
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    #plt.show()


#%%    
    
def profileHist(pos, data, fname, xlabel="x", ylabel="y", extent=[0.01,10,1e-7,0.5], nbins=200, logX=True, logY=True, meanLine=True, medianLine=True, ignoreZeros=True,  saveplot=False):
   
    print(extent)
    if logX:
        Xspace = np.logspace(np.log10(extent[0]), np.log10(extent[1]), nbins)
        extent[:2] = np.log10(extent[:2] )
    else:
        Xspace = np.linspace(extent[0],extent[1], nbins)
        
    if logY:
        Yspace = np.logspace(np.log10(extent[2]), np.log10(extent[3]), nbins)
        extent[2:] = np.log10(extent[2:] )
    else:
        Yspace = np.linspace(extent[2],extent[3], nbins)
    print(extent)
    rgas = pen.rval(Pos)
    histCounts,histBins1,histBins2  = np.histogram2d(Data, rgas, bins=(Yspace, Xspace))
    fig, ax1 = plt.subplots(figsize=(3.5,3.5), dpi=300)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.15)
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    plt.imshow(np.log10(histCounts), extent=extent, origin="lower")
    if ignoreZeros:
        getZeroCol = np.sum(histCounts, 0)==0
    else:
        getZeroCol = (np.sum(histCounts, 0)==-np.inf)

    if logX and logY:
        print("both log")
        xbins_c = 10**(np.log10(histBins2[:-1]) + np.diff(np.log10(histBins2))/2)
        ybins_c = 10**(np.log10(histBins1[:-1]) + np.diff(np.log10(histBins1))/2)
        if medianLine==True:
            medianPx = np.argmax(np.log10(histCounts), 0)
            plt.plot(np.log10(xbins_c[~getZeroCol]), np.log10(ybins_c[medianPx[~getZeroCol]]), lw=1, c="r")
        if meanLine==True:
            NperX = np.sum(histCounts, 0)[~getZeroCol]
            MeanY =np.sum(histCounts*ybins_c[:,np.newaxis], 0)[~getZeroCol]/NperX
            plt.plot(np.log10(xbins_c[~getZeroCol]), np.log10(MeanY), lw=1, c="y")
            
        plt.gca().set_aspect( (extent[1]-extent[0])/(extent[3]-extent[2]))
    
    if (logY and (logX==False)):
        print("ylog log")
        xbins_c = histBins2[:-1] + np.diff(histBins2)/2
        ybins_c = 10**(np.log10(histBins1[:-1]) + np.diff(np.log10(histBins1))/2)
        if medianLine:
            medianPx = np.argmax(np.log10(histCounts), 0)
            plt.plot(xbins_c[~getZeroCol], np.log10(ybins_c[medianPx[~getZeroCol]]), lw=3, c="r")
        if meanLine:
            NperX = np.sum(histCounts, 0)[~getZeroCol]
            MeanY =np.sum(histCounts*ybins_c[:,np.newaxis], 0)[~getZeroCol]/NperX
            plt.plot(xbins_c[~getZeroCol], np.log10(MeanY), lw=1, c="y")
        plt.gca().set_aspect( (extent[1]-extent[0])/(extent[3]-extent[2]))
        
    if (logY==False) and logX:
        print("xlog log")
        xbins_c = 10**(np.log10(histBins2[:-1]) + np.diff(np.log10(histBins2))/2)
        ybins_c = histBins1[:-1] + np.diff(histBins1)/2
        if medianLine:
            medianPx = np.argmax(np.log10(histCounts), 0)
            plt.plot(np.log10(xbins_c[~getZeroCol]), ybins_c[medianPx[~getZeroCol]], lw=3, c="r")
        if meanLine:
            NperX = np.sum(histCounts, 0)[~getZeroCol]
            MeanY =np.sum(histCounts*ybins_c[:,np.newaxis], 0)[~getZeroCol]/NperX
            plt.plot(np.log10(xbins_c[~getZeroCol]), MeanY, lw=1, c="y")
            
        plt.gca().set_aspect( (extent[1]-extent[0])/(extent[3]-extent[2]))
        
    if (logY==False) and (logX==False):
        print("both lin")
        xbins_c = histBins2[:-1] + np.diff(histBins2)/2
        ybins_c = histBins1[:-1] + np.diff(histBins1)/2
        if medianLine:
            medianPx = np.argmax(np.log10(histCounts), 0)
            plt.plot((xbins_c[~getZeroCol]), (ybins_c[medianPx[~getZeroCol]]), lw=3, c="r")
        if meanLine:
            NperX = np.sum(histCounts, 0)[~getZeroCol]
            MeanY =np.sum(histCounts*ybins_c[:,np.newaxis], 0)[~getZeroCol]/NperX
            plt.plot(xbins_c[~getZeroCol], MeanY, lw=1, c="y")
        
        plt.gca().set_aspect( (extent[1]-extent[0])/(extent[3]-extent[2]))    

    plt.colorbar()
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saveplot:
        plt.savefig(fname, format='png', dpi=300)
        plt.close()
    
"""
Just a test 
X = [True, False]
Y = [True, False]
counter = 0
for i in X:
    for j in Y:
        counter +=1
        profileHist(Pos, Data, fname="/home/mt/Penny/plot_test/hist_prof_{:03d}.png".format(counter), extent=[0.01,10,1e-7,0.5], logX=i, logY=j, saveplot=True)
"""    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    