import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Penny as pen
import time
#adapted from https://stackoverflow.com/questions/29832055/animated-subplots-using-matplotlib
"""
Animation example

Animations show a simple sccatter plot a density profile.
points on scatter plot are beig filtered out by density.

"""

start = time.time()
# loading the data
Data = pen.loadMost("../Data/snapshot_500", "gas")
rho = Data["rho"]
rho = np.log10(rho*pen.UnitDensity_in_cgs)
r = Data["rtot"]
pos= Data["pos"]
drho = np.linspace(rho.min(), rho.max(), 100)
def data_gen():
    cnt=0
    while drho[cnt] < rho.max():
        rho_curr = drho[cnt]
        pos_new = pos[rho>rho_curr]
        cnt+=1
        yield rho_curr, pos_new

data_gen.t = 0

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3.5), dpi=300)

# intialize three line objects (one in each axes)
line1, = ax1.plot([], [], ls="", marker=".", markersize=1.5, markeredgecolor="None", markerfacecolor="b", alpha=0.05)
line2, = ax2.plot([], [], ls="", marker=".", markersize=1.5, markeredgecolor="None", markerfacecolor="b", alpha=0.05)
line3, = ax2.plot([], [], lw=2, color="r")
line = [line1, line2, line3]


# Formating the fig
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)

ax1.set_aspect("equal")
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)
ax1.set_ylabel("x / pc")
ax1.set_xlabel("y / pc")
ax2.set_xlim(0,5)
ax2.set_ylim(rho.min(), rho.max())
ax2.set_ylabel("$ \log_{10}(\\rho$ /$g\,cm^{2})$")
ax2.set_xlabel("r / pc")


# initialize the data arrays 
x1data, x2data, x3data, y1data, y2data, y3data = [], [], [], [], [], []

def run(data):
    # update the data
    rho_curr, pos_new = data
    x1data = pos_new[:,0]
    x2data = r
    x3data = [0, 5]
    
    y1data = pos_new[:,1]
    y2data = rho
    y3data = [rho_curr, rho_curr]


    # axis limits checking. Same as before, just for both axes
    for ax in [ax1, ax2]:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_xlim()
        if rho_curr >= rho.max():
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.figure.canvas.draw()

    # update the data of both line objects
    line[0].set_data(x1data, y1data)
    line[1].set_data(x2data, y2data)
    line[2].set_data(x3data, y3data)

    return line

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
    repeat=False)
ani.save('../plot_test/basic_animation.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
print("Took me:", time.time() - start)
