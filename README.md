# Penny

Authors:
E-mail: 

## Summary 

A collection of often used basic analysis and visualization scripts for Gadget simulation snapshots, intended to minimize copy-pasting. Includes an example script.

## Requirements

numpy (ver ?)

matplotlib (ver ?)

pygadgetreader (https://github.com/jveitchmichaelis/pygadgetreader)

scipy (ver ?)

## Customization

User should define Code units in: gadget_helper/Units.py


## Installation

> python setup.py build     ## this builds the module  

> python setup.py install   ## this installs the module, may require sudo

## Usage

To load the tools, import the gadget_helper package:  
> import Penny as pen

Now you have access to various commands. Examples are given in gh_examples.py (this file contains only the script, no command definitions). 
Main commands are (parameters with * have default values so the command will run without providing them; it might not produce sensible results though):

**For reading things in**:  
___loader_f___(path, *partType, *wantedData) - reads in the snapshot at path, gets you the wantedData from particles of type partType. Run loader_f('dummy') to see the list of available partTypes and wantedDatas.
loadMost(path, *partType) reads in the snapshot at path, gets you the commonly-used data from particles of type partType; additionally, produces a few commonly used derivative quantities: rtot, vtot, vrad, vtan, angmom.

**Density map maker**:  
___make_Dmap_data___(path, extent, *depth, *quantity, *plane, *rezX, *rezY, *kernel) - reads in the snapshot at path, calculates the density map as an array of projected density values. The array has dimensions rezX*rezY and encompasses the region defined by extent. Only material within depth from the midplane is used. quantity can be density or temperature; plane can be XY, XZ, YZ; kernel should be the same as the one used in the simulation (default is "wendland2"). Returns both density and the snapshot time.

**Plotting**:  
___plotsnap___(rho, snaptime, extent, quantity, plane, fname) - plots a density map given by rho, adds a label of snaptime. extent is used to determine the axis ranges, quantity determines the label on the colourbar and the colour used for plotting (blue for density, red for temperature), plane determines axis labels. The plot is saved in fname (it should be the whole path to the file, unless you want the file in the same directory as the analysis tools).  
___plotprofile___(pos, data, snaptime, fname, *xlabel, *ylabel, *xmin, *xmax, *nbins, *logTrue, *meanLine, *medianLine) - plots a radial scatter plot and profile of data values, adds snaptime label, saves in fname. xlabel and ylabel are axis labels, xmin and xmax are horizontal axis ranges, nbins is number of bins for the calculation of mean/median line, logTrue determines whether the y axis should be logarithmic, meanLine overplots mean of the data in each bin, medianLine overplots median of the data.
