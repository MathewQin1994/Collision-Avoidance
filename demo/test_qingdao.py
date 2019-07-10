from src.planner.Astar_jit import DeliberativePlanner
from numpy import pi
from src.map.staticmap import Map, generate_do_trajectory
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import json

def test_qingdao():
    # 参数
    resolution_time = 10
    resolution_pos = 50
    default_speed = 4.114
    primitive_file_path = './case1/control_primitives.npy'
    e = 1.5

    # 起点、目标、地图
    s0 = tuple(np.array([13031, -7037, np.pi / 3, 0, 0], dtype=np.float64))
    sG = tuple(np.array((-2975, 3964, 0, 0, 0), dtype=np.float64))
    static_map = Map()
    with open("./case1/map_data1.json", 'r') as f:
        print("loading static_map")
        static_map.load_map(np.array(json.load(f)), 2, (-17000, -17000))

    es = [1, 1]
    for e in es:
        dp = DeliberativePlanner(
            static_map,
            resolution_pos,
            resolution_time,
            default_speed,
            primitive_file_path,
            e)
        start_time = time.time()
        tra = np.array(dp.start(s0, sG))
        print("runtime is {},closelist node number is {},trajectory total time is {}".format(
            time.time() - start_time, len(dp.closelist), tra[-1, -1]))
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
    fig.plot(s0[1], s0[0], 'ob', markersize=2)
    fig.plot(sG[1], sG[0], 'ob', markersize=2)
    fig.plot(tra[:, 1], tra[:, 0], "r")
    # for i in range(tra.shape[0]):
    #     if not np.isnan(tra[i,4]):
    #         fig.plot(tra[i,1],tra[i,0],"or",markersize=2)

    plt.show()