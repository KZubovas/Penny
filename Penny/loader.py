"""
Contains functions to load most used data
"""

import numpy as np
from pygadgetreader import *
#import dir_list as dl
from .basic import *


def loader_f(path, partType='None', wantedData=[]):
    if partType!='gas' and partType!='dm' and partType!='disk' and partType!='bulge' and partType!='star' and partType!='bndry' or len(wantedData)==0:
        print("""SUPPORTED PARTICLE TYPES:
   gas         - Type0: Gas
   dm          - Type1: Dark Matter
   disk        - Type2: Disk particles
   bulge       - Type3: Bulge particles
   star        - Type4: Star particles
   bndry       - Type5: Boundary particles
""")    
        print("""
---------------------
  -  STANDARD BLOCKS  -
  ---------------------
   pos         - (all)         Position data
   vel         - (all)         Velocity data code units
   pid         - (all)         Particle ids
   mass        - (all)         Particle masses
   u           - (gas)         Internal energy
   rho         - (gas)         Density
   ne          - (gas)         Number density of free electrons
   nh          - (gas)         Number density of neutral hydrogen
   hsml        - (gas)         Smoothing length of SPH particles
   sfr         - (gas)         Star formation rate in Msun/year
   age         - (stars)       Formation time of stellar particles
   z           - (gas & stars) Metallicty of gas/star particles (returns total Z)
   pot         - (all)         Potential of particles (if present in output)
""")

    else:
        DATA = {}
        for data in wantedData:
            DATA[data] = readsnap(path, data, partType, suppress=1)

    return DATA


def loadMost(path, partType="None"):
    if partType!='gas' and partType!='dm' and partType!='disk' and partType!='bulge' and partType!='star' and partType!='bndry':
        print("""SUPPORTED PARTICLE TYPES:
   gas         - Type0: Gas
   dm          - Type1: Dark Matter
   disk        - Type2: Disk particles
   bulge       - Type3: Bulge particles
   star        - Type4: Star particles
   bndry       - Type5: Boundary particles
""") 
        return 0
        
    if partType=="gas":
        wantedData=["pos", "vel", "pid", "mass", "u", "rho", "hsml"]
    else:
        wantedData=["pos", "vel", "pid", "mass"]
    data_in = loader_f(path, partType, wantedData)
    
    #extra parameters
    rtot = rval(data_in['pos'])
    vtot = rval(data_in['vel'])
    vrad = dotp(data_in['pos'],data_in['vel'])/rtot
    vtan = (vtot*vtot-vrad*vrad)**0.5
    angmom = getL(data_in['pos'],data_in['vel'])
    
    data_in['rtot'] = rtot
    data_in['vtot'] = vtot
    data_in['vrad'] = vrad
    data_in['vtan'] = vtan
    data_in['angmom'] = angmom
    
    return data_in
    