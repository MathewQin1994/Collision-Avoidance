import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from src.map.staticmap import Map, generate_do_trajectory
import matplotlib as mpl

colors = ['white', 'green', 'orange', 'blue', 'yellow', 'purple']
bounds = [0,1,2,3,4,5,6]

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# plt.imshow(my_array, interpolation='none', cmap=cmap, norm=norm)
static_map = Map()
static_map.load_map(np.loadtxt('../map/static_map1.txt',dtype=np.int8), 1)

fig = plt.gca()
extend = [
    static_map.offset[0],
    static_map.size[0] *
    static_map.resolution +
    static_map.offset[0],
    static_map.offset[1],
    static_map.size[1] *
    static_map.resolution +
    static_map.offset[1]]
fig.set_xlabel('E/m')
fig.set_ylabel('N/m')
mapplot = static_map.map.copy()
for i in range(mapplot.shape[0]):
    mapplot[i, :] = mapplot[i, :][::-1]
fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
plt.show()
