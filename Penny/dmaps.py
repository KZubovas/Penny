"""
Density map production
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from pygadgetreader import *
import matplotlib.cm as cm
import time
from tqdm import tqdm
import os


import Penny.loader as loader
import Penny.Units as unt
import Penny.basic as basic

import matplotlib.pyplot as plt

import scipy.spatial as scs


#this is the primary density map making routine for SPH (Gadget) simulation results

def make_Dmap_data(path, extent, depth=4, quantity='density', plane='XY', rezX=512, rezY=512, kernel="wendland2"):
    start = time.time()

#First we define the possible kernels and choose the one we will be using
    def wd2(u):
        return (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 + 4 * u) * 21 / 2 / np.pi / h[n]**3
    
    def wd4(u):
        return (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 + 6 * u + 35/3 * u**2 ) * 495 / 32 / np.pi
    
    def wd6(u):
        return (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 + 8 * u + 25 * u**2 + 32 * u**3) * 1365 / 64 / np.pi
    
    def cs(u):
        u05 = 1 - 6 * u**2 + 6 * u**3
        u1 = 2*(1 - u)**3
        u05[u>0.5] = u1[u>0.5]               
        return  u05 * 8 /np.pi / h[n]**3
    
    def ct(u):
        a = 1/3
        b = 1 +  6 * a**2 - 12 * a**3
        N = 8 / (np.pi * ( 6.4 * a**5 - 16 * a**6 + 1 ) )        
        ua = (-12 * a + 18 * a**2) * u + b
        u05 = 1 - 6 * u**2 + 6 * u**3
        u1 = 2*(1 - u)**3       
        ua[u>a] = u05[u>a]               
        ua[u>0.5] = u1[u>0.5] 
        return  ua * N / h[n]**3
    
    def hoct(u):
        N = 6.515
        P = -2.15
        Q = 0.918
        a = 0.75
        b = 0.5
        k =  0.214
        wk = np.zeros_like(u)
        
        uk = P * u + Q
        ub = (1 - u)**4 + (a - u)**4 + (b - u)**4
        ua = (1 - u)**4 + (a - u)**4
        u1 = (1 - u)**4
        
        wk = uk
        wk[u>k] = ub[u>k]
        wk[u>b] = ua[u>b]
        wk[u>a] = u1[u>a]
        
        return wk * N / h[n]**3
   
    kernel_list = {"wendland2":wd2, "wendland4":wd4, "wendland6":wd6, "CS":cs, "CT":ct, "HOCT":hoct}
    if kernel not in kernel_list.keys():
        print("Bad kernel choice, use oneof the following:")
        for i in kernel_list.keys():
            print("    "+i) 
            
        return 0
    print("Using ", kernel)  
    kf = kernel_list[kernel]

#We have chosen the kernel    

#Now we read the data from the snapshot
    import numpy as np
    snaptime = readhead(path,'time') * unt.UnitTime_in_s/ unt.year
    partlist = readhead(path,'npartTotal').tolist()
    print("")
    print("Reading snapshot ", path, ".")
    print("Snapshot time is ", snaptime, " yr.")
    print("Particle quantities of each type are: ", partlist, ".")
    if quantity == 'density':
        DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml'])
    if quantity == 'temperature':
        DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', 'u', 'rho'])        
        temp = DATA['u']*unt.u_to_temp_fac
        dens = DATA['rho']
    pos = DATA['pos']
    #rotate coordinates to get the plane I need
    if plane=='XY':
        xgas = pos[:,0]
        ygas = pos[:,1]
        zgas = pos[:,2]
    if plane=='YZ':
        xgas = pos[:,1]
        ygas = pos[:,2]
        zgas = pos[:,0]       
    if (plane=='ZX' or plane=='XZ'):
        xgas = pos[:,2]
        ygas = pos[:,0]
        zgas = pos[:,1]
        
    if (plane!='XY' and plane!='YZ' and plane!='XZ' and plane!='ZX'):
        print("Error: wrong plane statement.")
        return 0
    
    mgas = DATA['mass']
    hsml = DATA['hsml']

    #choose the size of the plotting region
    XL = abs(extent[1] - extent[0])
    YL = abs(extent[3] - extent[2])
    rho_val = np.zeros([rezY,rezX])
    darea =  (XL)*(YL)/rezX/rezY    
    xgn = xgas - extent[0]
    ygn = ygas - extent[2]
        
    #pixel size
    pixelsizeY = YL / rezY
    pixelsizeX = XL / rezX
    
    #min/max smoothing lengths
    hmin = 1.001 * (pixelsizeY**2 + pixelsizeX**2)**0.5 / 2.
    hmax = (pixelsizeY**2 + pixelsizeX**2)**0.5 * 3000
     
    #Select only the particles that are relevant to the plotting region
    a = np.array(xgn - hsml <= XL) 
    b = np.array(ygn - hsml <= YL)
    c = np.array(xgn + hsml >= 0) 
    d = np.array(ygn + hsml >= 0)
    e = np.array(np.abs(zgas) <= depth)
    condition_a = a&b&c&d&e
    
    h = hsml.copy()
    h[h < hmin] = hmin
    h[h >= hmax] = hmax
    
    x = (np.floor(xgn / pixelsizeX) + 0.5) * pixelsizeX
    y = (np.floor(ygn / pixelsizeY) + 0.5) * pixelsizeY
    npx = np.floor(h / pixelsizeX)
    npy = np.floor(h / pixelsizeY)
    print('Starting density calculation:')
    for n in tqdm(range(len(xgn))):
        if (condition_a[n] == True):
            dx = np.arange(-npx[n], npx[n]+1,1)
            dy = np.arange(-npy[n], npy[n]+1,1)
                    
            xx = x[n] + pixelsizeX * dx - xgn[n]
            yy = y[n] + pixelsizeY * dy - ygn[n]
            
            #2D matrix with squared distances
            r2 = (yy * yy)[:, np.newaxis] + xx * xx
    
            
            u = np.sqrt(r2) / h[n]
            
            #Kernel value
            wk = kf( u )
            wk[u*h[n] > h[n]] = 0 
            summ = np.sum( wk )
            if summ > 10e-10:
                xxx = xx + xgn[n]
                yyy = yy + ygn[n]
    
                #indexai realus
                maskX = (xxx < XL) & (xxx > 0)
                maskY =  (yyy < YL) &  (yyy > 0)
                masksq = maskY[:,np.newaxis] * maskX
                i = xxx[maskX] / pixelsizeX
                j = yyy[maskY] / pixelsizeY
                i = np.array(np.floor(i), dtype=int)
                j = np.array(np.floor(j), dtype=int)
                 
                if quantity == 'density':
                    rho_val[j[:,np.newaxis], i] += mgas[n]*wk[masksq].reshape([len(j), len(i)])/summ
                if quantity == 'temperature':
                    rho_val[j[:,np.newaxis], i] += mgas[n]* temp[n]/dens[n] * wk[masksq].reshape([len(j), len(i)])/summ
    rho_val = rho_val / darea # *Unit Column Density
    print(quantity,' map done [in code units]')
    print('It took', time.time()-start, 'seconds.')
    return rho_val, snaptime




#Fixed to give correct average velocity/temperature etc. Use rho_val/delta_l
def make_Dmap_data_mean(path, extent, depth=4, quantity='density', plane='XY',
                   rezX=512, rezY=512, kernel="wendland2", mock_data=False, use_wedge_cut=False, cut_angle=np.pi/128):
    start = time.time()

#First we define the possible kernels and choose the one we will be using
    def wd2(u):
        return (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 + 4 * u) * 21 / 2 / np.pi / h[n]**3
    
   
    kernel_list = {"wendland2":wd2}
    if kernel not in kernel_list.keys():
        print("Bad kernel choice, use oneof the following:")
        for i in kernel_list.keys():
            print("    "+i) 
            
        return 0
    print("Using ", kernel)  
    kf = kernel_list[kernel]

#We have chosen the kernel    

#Now we read the data from the snapshot
    import numpy as np
    snaptime = readhead(path,'time') * unt.UnitTime_in_s/ unt.year
    partlist = readhead(path,'npartTotal').tolist()
    print("")
    print("Reading snapshot ", path, ".")
    print("Snapshot time is ", snaptime, " yr.")
    print("Particle quantities of each type are: ", partlist, ".")
    if mock_data==False:
        if quantity == 'density':
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml'])
        if quantity == 'temperature':
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', 'u',"vel", 'rho'])        
            temp = DATA['u']*unt.u_to_temp_fac
            dens = DATA['rho']
        if quantity == "vx":
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', "vel", 'rho'])
            vel = DATA["vel"]
            dens = DATA['rho']
        if quantity == "vy":
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', "vel", 'rho'])
            vel = DATA["vel"]
            dens = DATA['rho']
        if quantity == "vz":
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', "vel", 'rho'])
            vel = DATA["vel"]
            dens = DATA['rho']
        if quantity =="vr":
            DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', "vel", 'rho'])
            vel = DATA["vel"]
            dens = DATA['rho']
    else:
        temp = mock_data['u']*unt.u_to_temp_fac
        pos = mock_data['pos']
        mgas = mock_data['mass']
        vel = mock_data['vel']
        dens = mock_data['rho']
        hsml = mock_data['hsml']
        
    if mock_data==False:
        pos = DATA['pos']
    else:
        pos = mock_data['pos']
    print(pos.shape)
    #rotate coordinates to get the plane I need
    if plane=='XY':
        xgas = pos[:,0]
        ygas = pos[:,1]
        zgas = pos[:,2]
        if quantity == 'vx' or quantity == 'vy' or quantity == 'vz' or quantity == 'vr':
            vely = vel[:,1]
            velx = vel[:,0]
            vely = vel[:,2]
    if plane=='YZ':
        xgas = pos[:,1]
        ygas = pos[:,2]
        zgas = pos[:,0]       
        if quantity == 'vx' or quantity == 'vy' or quantity == 'vz' or quantity == 'vr':
            velx = vel[:,1]
            vely = vel[:,2]
            velz = vel[:,0]
    if (plane=='ZX' or plane=='XZ'):
        xgas = pos[:,2]
        ygas = pos[:,0]
        zgas = pos[:,1]
        if quantity == 'vx' or quantity == 'vy' or quantity == 'vz' or quantity == 'vr':
            velx = vel[:,2]
            vely = vel[:,0]
            velz = vel[:,1]
        
    if quantity == "vz":
        vel = velz
    if quantity == "vy":  
        vel = vely
    if quantity =="vx":
        vel = velx
    if quantity =="vr":
        vel = basic.ToSph(vel)[:,0]

    sphPos = basic.ToSph( np.array([pos[:,2], pos[:,0], pos[:,1]]).T )

    if (plane!='XY' and plane!='YZ' and plane!='XZ' and plane!='ZX'):
        print("Error: wrong plane statement.")
        return 0
    if mock_data==False:
        mgas = DATA['mass']
        hsml = DATA['hsml']

    #choose the size of the plotting region
    XL = abs(extent[1] - extent[0])
    YL = abs(extent[3] - extent[2])
    rho_val = np.zeros([rezY,rezX])
    darea =  (XL)*(YL)/rezX/rezY    
    xgn = xgas - extent[0]
    ygn = ygas - extent[2]
        
    #pixel size
    pixelsizeY = YL / rezY
    pixelsizeX = XL / rezX
    #min/max smoothing lengths
    hmin = 1.001 * (pixelsizeY**2 + pixelsizeX**2)**0.5 / 2.
    hmax = (pixelsizeY**2 + pixelsizeX**2)**0.5 * 3000
     
    #Select only the particles that are relevant to the plotting region
    a = np.array(xgn - hsml <= XL) 
    b = np.array(ygn - hsml <= YL)
    c = np.array(xgn + hsml >= 0) 
    d = np.array(ygn + hsml >= 0)
    e = np.array(np.abs(zgas) <= depth)
    
    if use_wedge_cut:
        f= np.array(np.abs(np.abs(sphPos[:,1])-np.pi/2) < cut_angle )
        print("Using wegge instead of depth; Current cut_angle=",cut_angle)
        condition_a = a&b&c&d&f
    else:
        print("Using depth instead of depth; Current depth=",depth)
        condition_a = a&b&c&d&e
    h = hsml.copy()
    h[h < hmin] = hmin
    h[h >= hmax] = hmax
    delta_l = np.zeros_like(rho_val)
    x = (np.floor(xgn / pixelsizeX) + 0.5) * pixelsizeX
    y = (np.floor(ygn / pixelsizeY) + 0.5) * pixelsizeY
    npx = np.floor(h / pixelsizeX)
    npy = np.floor(h / pixelsizeY)
    print('Starting density calculation:')
    for n in tqdm(range(len(xgn))):
        if (condition_a[n] == True):
            #print(n)
            dx = np.arange(-npx[n], npx[n]+1,1)
            dy = np.arange(-npy[n], npy[n]+1,1)
                    
            xx = x[n] + pixelsizeX * dx - xgn[n]
            yy = y[n] + pixelsizeY * dy - ygn[n]
            
            #2D matrix with squared distances
            r2 = (yy * yy)[:,np.newaxis] + xx * xx
            corrl =  2 * (h[n]**2 - r2)**0.5# / 2 / h[n]
            u = np.sqrt(r2) / h[n]
            
            #Kernel value
            wk = kf(u)
            wk[u*h[n] > h[n]] = 0 
            corrl[u*h[n] > h[n]] = 0 
            summ = np.sum( wk )
            #print(wk.shape, corrl.shape)
            #crash
            if summ > 10e-10:
                xxx = xx + xgn[n]
                yyy = yy + ygn[n]
                    
                #indexai realus
                maskX = (xxx < XL) & (xxx > 0)
                maskY =  (yyy < YL) &  (yyy > 0)
                masksq = maskY[:,np.newaxis] * maskX
                i = xxx[maskX] / pixelsizeX
                j = yyy[maskY] / pixelsizeY
                i = np.array(np.floor(i), dtype=int)
                j = np.array(np.floor(j), dtype=int)
                 
                if quantity == 'density':
                    rho_val[j[:,np.newaxis], i] += mgas[n]*wk[masksq].reshape([len(j), len(i)])/summ
                    delta_l[j[:,np.newaxis], i] += corrl[masksq].reshape([len(j), len(i)])
                if quantity == 'temperature':
                    rho_val[j[:,np.newaxis], i] += mgas[n]* temp[n]/dens[n] * wk[masksq].reshape([len(j), len(i)])/summ
                    delta_l[j[:,np.newaxis], i] += corrl[masksq].reshape([len(j), len(i)])
                if quantity == 'vx' or quantity == 'vy' or quantity == 'vz' or quantity == 'vr':
                    rho_val[j[:,np.newaxis], i] += mgas[n]* vel[n]/dens[n] * wk[masksq].reshape([len(j), len(i)])/summ
                    delta_l[j[:,np.newaxis], i] += corrl[masksq].reshape([len(j), len(i)])
                    
    rho_val = rho_val / darea # *Unit Column Density
    print(quantity,' map done [in code units]')
    print('It took', time.time()-start, 'seconds.')
    return rho_val, snaptime, delta_l

#this is the primary density map making routine for moving-mesh (Arepo) simulation results

def make_Dmap_data_Tree(path, quantity='density', extent=[-5, 5, -5., 5.,-1.25, 1.25], plane='XY', rezX=512, rezY=512, rezZ=124, hsml_cut=True):
    print("here be dragons")
    start = time.time()
    
    print("")
    snaptime = readhead(path,'time') * unt.UnitTime_in_s/ unt.year
    print("Snapshot time is ", snaptime, " yr.")
    if quantity == 'density':
        DATA[quantity] = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml','rho'])
    if quantity == 'temperature':
        DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', 'u', 'rho'])
        temp = DATA['u']*unt.u_to_temp_fac
        dens = DATA['rho']
    if (quantity != 'temperature') and (quantity != 'density'):
        print("using "+quantity+" as quantity.")
        DATA = loader.loader_f(path, partType='gas',wantedData=['pos','mass', 'hsml', 'u', quantity])
        
    pos = DATA['pos'].astype(np.float)
    if plane=='XY':
        xgas = pos[:,0]
        ygas = pos[:,1]
        zgas = pos[:,2]
    if plane=='YZ':
        xgas = pos[:,2]
        ygas = pos[:,0]
        zgas = pos[:,1]       
    if (plane=='ZX' or plane=='XZ'):
        xgas = pos[:,1]
        ygas = pos[:,2]
        zgas = pos[:,0]
    rho = DATA[quantity].astype(np.float)
    if hsml_cut:
        h = DATA["hsml"].astype(np.float)
    ## extent given in code units
    XX, YY = np.meshgrid(np.linspace(extent[0],extent[1],rezX),np.linspace(extent[2],extent[3],rezY)) ## Būtų greičiau, perduot mesh grid, o ne generuot iš skyros kiekvieną kart
    XX = XX.reshape(-1).astype(np.float)
    YY = YY.reshape(-1).astype(np.float)
    ZZ = np.linspace(extent[4],extent[5],rezZ)
    Rho = np.zeros([rezY, rezX])
    Tree = scs.cKDTree(pos)
    for z in ZZ:
        zzz = (z * np.ones_like(XX)).astype(np.float)
        R, IND = Tree.query(np.array([XX,YY,zzz]).T, k=1)
        rho1 = rho[IND.astype(int)]#.reshape(rezy,rezx) * gh.UnitDensity_in_cgs * L  # * gh.UnitMass_in_g / A
        if hsml_cut:
            h1 = h[IND.astype(int)]
            rho1[R>h1] = 0
        rho1=rho1.reshape(rezY,rezX)
                          
        Rho += rho1
        print( '\rAt z level {:3f}'.format(z), end='',flush=True)
    print('Rho map done [in code units]')
    print('It took', time.time()-start, 'seconds.')
    return Rho


#this creates a 3D cube of voxels with density values; use at your own peril!

def make_Dmap3D_data(path, extent=[-5, 5, -5., 5., -5., 5.], rezX=512, rezY=512, rezZ=512):
    start = time.time()
    DATA = loader.loader_f(path, partType='gas', wantedData=['pos','mass', 'hsml'])
    pos = DATA['pos'].astype('f')
    mgas = DATA['mass'].astype('f')
   #LEN = mgas.shape[0]
    hsml = DATA['hsml'][:].astype('f')
    #Split DATA
    mgas = mgas[:].astype('f')
    
    xgas = pos[:,0].astype('f')
    ygas = pos[:,1].astype('f')
    zgas = pos[:,2].astype('f')

    XL = abs(extent[0] - extent[1])
    YL = abs(extent[2] - extent[3])
    ZL = abs(extent[4] - extent[5])
    #apibreziu tankio masyva
    rho_val = np.zeros([rezY,rezX,rezZ]).astype('f')
    #normavimo konstanta
    darea =  (XL)*(YL)*(ZL)/rezX/rezY/rezZ    
    #Erdvė turi būti teigiama, pastumti gali ir nereikėti
    xgn = xgas + extent[1]
    ygn = ygas + extent[3]
    zgn = zgas + extent[3]
    
    
    #pixelio dydis
    pixelsizeY = YL / rezY
    pixelsizeX = XL / rezX
    pixelsizeZ = ZL / rezZ
    #glotninimo ilgio  min/max
    hmin = 1.001 * (pixelsizeY**2 + pixelsizeX**2 + pixelsizeZ**2)**0.5 / 2.
    hmax = (pixelsizeY**2 + pixelsizeX**2 + pixelsizeZ**2)**0.5 * 30
    #Dalelės tinkamumo sąlygos
    a = np.array(xgn - hsml <= XL) 
    b = np.array(ygn - hsml <= YL)
    f = np.array(zgn - hsml <= ZL)
    c = np.array(xgn + hsml >= 0) 
    e = np.array(ygn + hsml >=0)
    g = np.array(zgn + hsml >=0)
    
    Az = np.array(zgn + hsml >=0)&np.array(zgn<=0)
    Ax = np.array(xgn + hsml >=0)&np.array(xgn<=0)
    Ay = np.array(ygn + hsml >=0)&np.array(ygn<=0)
    
    Vz = np.array(zgn - hsml >=ZL)&np.array(zgn>=ZL)
    Vx = np.array(xgn - hsml >=XL)&np.array(xgn>=XL)
    Vy = np.array(ygn - hsml >=YL)&np.array(ygn>=YL)
    #d = np.array(np.abs(zgas) <= 4)
    #condition_a = a&b&c&d&e
    condition_a = a&b&c&e&f&g#Vz|Ay|Ax|Az|Vx|Vy|Vz
    
    h = hsml.copy()
    h[h < hmin] = hmin
    h[h >= hmax] = hmax
    
    x = (np.floor(xgn / pixelsizeX) + 0.5) * pixelsizeX
    y = (np.floor(ygn / pixelsizeY) + 0.5) * pixelsizeY
    z = (np.floor(zgn / pixelsizeZ) + 0.5) * pixelsizeZ
    npx = np.floor(h / pixelsizeX)
    npy = np.floor(h / pixelsizeY)
    npz = np.floor(h / pixelsizeZ)
    print('\n######################\nStarting density calculation:')
    for n in tqdm(range(len(xgn))):
        if (condition_a[n] == True):
            dx = np.arange(-npx[n], npx[n]+1,1)
            dy = np.arange(-npy[n], npy[n]+1,1)
            dz = np.arange(-npz[n], npz[n]+1,1)
            
            xx = x[n] + pixelsizeX * dx - xgn[n]
            yy = y[n] + pixelsizeY * dy - ygn[n]
            zz = z[n] + pixelsizeZ * dz - zgn[n]
            
            #2D matrica su spindulio kvadratais
            r2 = (yy * yy)[:,np.newaxis] + xx * xx                          
            r3 = r2[:,:,np.newaxis] + (zz * zz)#[:,np.newaxis]
            r = np.sqrt(r3)
            
            u = np.sqrt(r3) / h[n]
            
            #Wendland kažkuris kernelis
            wk = (1 - u) * (1 - u) * (1 - u) * (1 - u) * (1 + 4 * u)
            wk[u*h[n] > h[n]] = 0 
            summ = np.sum( wk )
            if summ > 10e-10:
    
                xxx = xx + xgn[n]
                yyy = yy + ygn[n]
                zzz = zz + zgn[n]
                
                #indexai realus
                maskX = (xxx < XL) & (xxx > 0)
                maskY =  (yyy < YL) &  (yyy > 0)
                maskZ =  (zzz < ZL) &  (zzz > 0)
                
                masksq = maskY[:,np.newaxis] * maskX
                mask3d = masksq[:,:, np.newaxis] * maskZ[:]
                mask3d = np.transpose(mask3d, axes=(2,0,1))
                
                i = xxx[maskX] / pixelsizeX
                j = yyy[maskY] / pixelsizeY
                k = zzz[maskZ] / pixelsizeZ
                
                i = np.array(np.floor(i), dtype=int)
                j = np.array(np.floor(j), dtype=int)
                k = np.array(np.floor(k), dtype=int)
                #rho_val[j[:,np.newaxis], i] += mgas[n]*wk[masksq].reshape([len(j), len(i)])/summ
                rho_val[k[:,np.newaxis,np.newaxis], j[:,np.newaxis], i] += mgas[n]*wk[mask3d].reshape([len(k),len(j), len(i)]) / summ
                #rho_val[0,1:5,1:5] += 0.2
            del dx, dy, dz, xx, yy, zz, r2, r3, r, u, wk, summ, xxx, yyy, zzz, maskX, maskY, maskZ, masksq, mask3d, i, j, k
    rho_val = rho_val  / darea # Dauginti i6 unit column density
    print('It took', time.time()-start, 'seconds.')
    return rho_val

    
    
    
    