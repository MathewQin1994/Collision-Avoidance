import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from src.map.staticmap import Map, generate_do_trajectory

static_map = Map()
static_map.load_map(np.loadtxt('o1.txt',dtype=np.int8), 1)

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
fig.imshow(mapplot.T, extent=extend)
plt.show()
