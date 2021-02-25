# ## Model Code to real Units

## CND Cloud model units
UnitLength_in_cm     =   3.086 * 10**18
UnitMass_in_g        =   1.989 * 10**33 * 5. *10**6
boxsize = 10
UnitVelocity_in_cm_per_s = (6.67*10**(-8) * UnitMass_in_g/UnitLength_in_cm)**(1/2);
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs =  UnitMass_in_g * UnitLength_in_cm**2 / UnitTime_in_s**2
UnitDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**3
UnitColumnDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**2
## KZ UNITS
#UnitLength_in_cm     =   3.086 * 10**21 #10**18
#UnitMass_in_g        =   1.989 * 10**33 * 2 * 10**10 #* 5. *10**6
#boxsize = 10
#UnitVelocity_in_cm_per_s = (6.67*10**(-8) * UnitMass_in_g/UnitLength_in_cm)**(1/2)
#UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
#UnitEnergy_in_cgs =  UnitMass_in_g * UnitLength_in_cm**2 / UnitTime_in_s**2
#UnitDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**3
#UnitColumnDensity_in_cgs = UnitMass_in_g/UnitLength_in_cm**2
#M_s = 1.9891 * 10**33 #g
#c = 3*10**10 #cm/s
#yr_in_s = 31556926
#Le = 1.3*10**38*(4*1e6)  #erg/s 5.2 * 10^44
#Me = Le / 0.1 / c**2
r_s = (2*6.67259*1e-8*1.99*4*1e6*1e33) / (2.9979*1e10)**2 ## cm
lu = 11792560000.000002 #m
tu = 55.62914557076172
mu = 7.956000000000001e+36 
A = 5.67e-5 #erg/cm^2/K^4/s
#print(Me*yr_in_s/M_s)
#:print(Me)

#constants

#     Natural constants in CGS

GG  = 6.672e-8       # Gravitational constant
mp  = 1.6726e-24     # Mass of proton          [g]
me  = 9.1095e-28     # Mass of electron        [g]
kk  = 1.3807e-16     # Bolzmann's constant     [erg/K]
hh  = 6.6262e-27     # Planck's constant       [erg.s]
ee  = 4.8032e-10     # Unit charge             
cc  = 2.9979e10      # Light speed             [cm/s]
st  = 6.6524e-25     # Thompson cross-section  [cm^2]
ss  = 5.6703e-5      # Stefan-Boltzmann const  [erg/cm^2/K^4/s]
aa  = 7.5657e-15     # 4 ss / cc               [erg/cm^3/K^4]
#                                                              
#     Gas constants                                            
#                                                              
muh2= 2.3000e0       # Mean molec weight H2+He+Metals
mu_ion = 0.63       #fully ionized solar metallicity
mu_mol = 2.54       #molecular solar metallicity
#                                                              
#     Alternative units                                        
#                                                              
ev  = 1.6022e-12     # Electronvolt            [erg]
kev = 1.6022e-9      # Kilo electronvolt       [erg]
micr= 1.0e-4          # Micron                  [cm]
km  = 1.0e5           # Kilometer               [cm]
angs= 1.0e-8          # Angstroem               [cm]
#                                                              
#     Astronomy constants                                      
#                                                              
LS  = 3.8525e33      # Solar luminosity        [erg/s]
RS  = 6.96e10        # Solar radius            [cm]
MS  = 1.99e33        # Solar mass              [g]
TS  = 5.78e3         # Solar temperature       [K]
AU  = 1.496e13       # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
#                                                              
#     Time units                                               
#                                                              
year= 3.1536e7       # Year                    [s]
hour= 3.6000e3       # Hour                    [s]
day = 8.64e4         # Day                     [s]
#
#     Math constants
#
pi  = 3.1415926535897932385 
