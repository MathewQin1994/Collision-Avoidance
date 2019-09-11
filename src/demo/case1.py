# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
from src.planner.Astar_jit import DeliberativePlanner
from numpy import pi
from src.map.staticmap import Map, generate_do_trajectory
from src.demo.simulation import simulation
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging



if __name__=="__main__":

    # 地图、起点、目标
    static_map = Map()
    static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), 1)
    s0 = tuple(np.array((174, 201, 0.86-pi, 0.8, 10), dtype=np.float64))
    # s0 = tuple(np.array((190, 1, 1.57, 0.8, 0), dtype=np.float64))
    sG = tuple(np.array((41, 71, pi, 0.8, 0), dtype=np.float64))
    # sG = tuple(np.array((55, 180, pi, 0.8, 0), dtype=np.float64))


    fig = plt.gca()
    extend = [
        static_map.offset[0],
        static_map.size[0] *
        static_map.resolution +
        static_map.offset[0]-1,
        static_map.offset[1],
        static_map.size[1] *
        static_map.resolution +
        static_map.offset[1]-1]
    mapplot = static_map.map.copy()
    static_map.expand(2)
    for i in range(mapplot.shape[0]):
        mapplot[i, :] = mapplot[i, :][::-1]
    fig.imshow(mapplot.T, extent=extend)
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    fig.plot(sG[1], sG[0], "ob", markersize=5)
    # plt.show()
    # 本船参数和规划器
    resolution_time = 1
    resolution_pos = 1
    default_speed = 0.8
    primitive_file_path = '../primitive/control_primitives.npy'
    e = 1.2
    dp = DeliberativePlanner(
        static_map,
        resolution_pos,
        resolution_time,
        default_speed,
        primitive_file_path,
        e)

    # 他船参数和规划器
    do_s0=dict()
    do_dp = dict()
    do_tra_true = dict()
    do_goal=dict()

    do_s0['1']=(190, 1, 1.57, 0.8, 0)
    do_goal['1'] = (53, 121)
    do_s0['2']=(35, 114, 0, 0.8, 0)
    do_goal['2'] = (160, 177)
    do_s0['3']=(159, 155, pi, 0.4, 0)
    do_goal['3'] = (99, 93)
    # do_s0['4']=(120, 90, pi/2, 0.4, 0)
    # do_goal['4'] = (156, 173)

    for key in do_s0:
        do_dp[key] = DeliberativePlanner(
            static_map,
            resolution_pos,
            resolution_time,
            do_s0[key][3],
            '../primitive/control_primitives{}.npy'.format(do_s0[key][3]),
            e)
        do_tra_true[key]=np.array(do_dp[key].start(do_s0[key],do_goal[key]))


    simulation(s0, sG, dp, do_dp,fig,do_tra_true,do_goal,predict_time=10)
    # simulation(s0, sG, dp, do_dp, fig)
    # tra=dp.start(s0,sG)
    plt.show()