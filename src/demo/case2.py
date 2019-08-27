from src.planner.Astar_jit import DeliberativePlanner
from numpy import pi
from src.map.staticmap import Map, generate_do_trajectory
from src.demo.simulation import simulation
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging


#
#
# s0 = tuple(np.array((10, 50, 0, 0.8, 10), dtype=np.float64))
# sG = tuple(np.array((95, 50, pi, 0.8, 0), dtype=np.float64))
# fig.plot(sG[1], sG[0], "ob", markersize=5)

if __name__=="__main__":
    # 参数
    resolution_time = 1
    resolution_pos = 1
    default_speed = 0.8
    primitive_file_path = '../primitive/control_primitives.npy'
    e = 1.2

    # 地图、起点、目标
    static_map = Map()
    static_map.load_map(np.loadtxt('../map/static_map2.txt', dtype=np.int8), 1)
    s0 = tuple(np.array((193, 166, 0, 0.8, 10), dtype=np.float64))
    sG = tuple(np.array((74, 222, pi, 0.8, 0), dtype=np.float64))

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
    for i in range(mapplot.shape[0]):
        mapplot[i, :] = mapplot[i, :][::-1]
    fig.imshow(mapplot.T, extent=extend)
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    # 动态障碍物真实轨迹
    # case1
    # do_tra_true=dict()
    # do_tra_true['1']=generate_do_trajectory(50,95,-pi/2,0.8,200)
    # do_tra_true['2']=generate_do_trajectory(95,95,-pi*3/4,0.8,200)
    # sg1 = (50, 0)
    # sg2 = (0, 0)

    # case2
    do_tra_true = dict()
    do_goal=dict()
    do_tra_true['1'] = generate_do_trajectory(84, 94, 0.86, 0.8, 200)
    # do_tra_true['2'] = generate_do_trajectory(85, 85, -pi * 3 / 4, 0.8, 200)
    do_goal['1']=(185, 213)
    # do_goal['2']=(0,0)

    # 规划器
    dp = DeliberativePlanner(
        static_map,
        resolution_pos,
        resolution_time,
        default_speed,
        primitive_file_path,
        e)
    do_dp = dict()
    for key in do_tra_true:
        do_dp[key] = DeliberativePlanner(
            static_map,
            resolution_pos,
            resolution_time,
            default_speed,
            primitive_file_path,
            e)

    # simulation(s0, sG, dp, do_dp,fig,do_tra_true,do_goal)
    simulation(s0, sG, dp, do_dp, fig)
    # tra=dp.start(s0,sG)
    plt.show()