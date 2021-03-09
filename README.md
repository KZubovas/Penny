---


---

<h1 id="penny">Penny</h1>
<p>Authors: M. Tartėnas, K. Zubovas<br>
E-mail: <a href="mailto:matas.tartenas@ftmc.lt">matas.tartenas@ftmc.lt</a>, <a href="mailto:kastytis.zubovas@ftmc.lt">kastytis.zubovas@ftmc.lt</a></p>
<h2 id="summary">Summary</h2>
<p>A collection of often used basic analysis and visualization scripts for <em>Gadget</em> simulation snapshots, intended to minimize copy-pasting. Includes an example script.</p>
<h2 id="requirements">Requirements</h2>
<p>numpy<br>
matplotlib<br>
<a href="https://github.com/jveitchmichaelis/pygadgetreader">pygadgetreader</a><br>
scipy</p>
<h2 id="customization">Customization</h2>
<p>User should define Code units in: Penny/Units.py. Do this <em>before</em> building and installing the package. In general, you should only need to define the unit length and mass based on what is used in your simulation, the rest of the units are calculated based on these, assuming G = 1 in code units.</p>
<h2 id="installation">Installation</h2>
<blockquote>
<p>python <a href="http://setup.py">setup.py</a> build     ## this builds the module</p>
</blockquote>
<blockquote>
<p>python <a href="http://setup.py">setup.py</a> install   ## this installs the module, may require sudo</p>
</blockquote>
<h2 id="usage">Usage</h2>
<p>To load the tools, import the Penny package:</p>
<blockquote>
<p>import Penny as pen</p>
</blockquote>
<p>Now you have access to various commands. Examples are given in example_scripts/density_plotter.py and example_scripts/some_plotting_examples.py (these files contain only the script, no command definitions).</p>
<p>Main commands are (parameters with * have default values so the command will run without providing them; it might not produce sensible results though):</p>
<p><strong>For reading things in</strong>:</p>
<ul>
<li>
<p><em>loader_f(path, *partType, *wantedData)</em> - reads in the snapshot at path, gets you the wantedData from particles of type partType. Run loader_f(‘dummy’) to see the list of available <em>*partTypes</em>  and  <em>*wantedData</em>.</p>
</li>
<li>
<p><em>loadMost(path, *partType)</em> reads in the snapshot at path, gets you the commonly-used data from particles of type partType; additionally, produces a few commonly used derivative quantities: rtot, vtot, vrad, vtan, angmom.</p>
</li>
</ul>
<p><strong>Density map generation</strong>:</p>
<ul>
<li>
<p><em>make_Dmap_data(path, extent, *depth, *quantity, *plane, *rezX, *rezY, *kernel)</em> - reads in the snapshot at path, calculates the density map as an array of projected density values. The array has dimensions rezX*rezY and encompasses the region defined by extent. Only material within depth from the midplane is used. quantity can be density or temperature; plane can be XY, XZ, YZ; kernel should be the same as the one used in the simulation (default is “wendland2”). Returns both density and the snapshot time.</p>
</li>
<li>
<p><em>make_Dmap_data_Tree(path, *quantity, extent, *plane, *rezX, *rezY, *rezZ, *hsml_cut)</em> -  reads in the snapshot at path, calculates the density map of any select quantity. Values for a given pixel are determined by the closest particle using scipy.spatial.cKDtree. This is performed sequentially adding together a given *rezZ number of layers. Preffered method for moving mesh codes.</p>
</li>
<li>
<p><em>make_Dmap3D_data(path, *extent, *rezX, *rezY, *rezZ)</em> - same as <em>make_Dmap_data()</em>, but makes a data cube (Ex. could be used to generate slightly more accurate column density maps). <em>make_Dmap_data()</em> is recommened instead, as it takes much less time, but this could be usefull in specific cases.</p>
</li>
</ul>
<p><strong>Plotting</strong>:</p>
<ul>
<li>
<p><em>plotsnap(rho, snaptime, extent, quantity, plane, fname)</em> - plots a density map given by rho, adds a label of snaptime. extent is used to determine the axis ranges, quantity determines the label on the colourbar and the colour used for plotting (blue for density, red for temperature), plane determines axis labels. The plot is saved in fname (it should be the whole path to the file, unless you want the file in the same directory as the analysis tools).</p>
</li>
<li>
<p><em>plotprofile(pos, data, snaptime, fname, *xlabel, *ylabel, *xmin, *xmax, *nbins, *logTrue, *meanLine, *medianLine)</em> - plots a radial scatter plot and profile of data values, adds snaptime label, saves in fname. xlabel and ylabel are axis labels, xmin and xmax are horizontal axis ranges, nbins is number of bins for the calculation of mean/median line, logTrue determines whether the y axis should be logarithmic, meanLine overplots mean of the data in each bin, medianLine overplots median of the data.</p>
</li>
</ul>
<p><strong>General use</strong>:</p>
<ul>
<li>
<p><em>rval(arr)</em> - get radial values of a given shape(m,n) <em>arr</em> (Ex. rgas = rval(pos), where rgas is the radial distance from centre of the system and pos is the postition of many particles in space)</p>
</li>
<li>
<p><em>dotp(arr)</em> - dot product of two vectors</p>
</li>
<li>
<p><em>getL(arr1,arr2)</em> - vector product of two vectors</p>
</li>
<li>
<p><em>ToSph(pos)</em> - Tranformation function from Cartesian to Spherical coordinates</p>
</li>
<li>
<p><em>CartRot(pos, theta, phi )</em> - rotation of Cartesian coordinates</p>
</li>
<li>
<p><em>scatterPlot(x, y, *alpha, *marker)</em> - basic scatter plot; faster, but less functional than <em>plt.scatter()</em></p>
</li>
</ul>
<p><strong>Example scripts</strong></p>
<ul>
<li>
<p>density_plotter.py - simple example of plotting multiple density maps using <em>make_Dmap_data()</em> and <em>plotsnap()</em></p>
</li>
<li>
<p>some_plotting_examples.py - example of using many of the functions provided</p>
</li>
</ul>

