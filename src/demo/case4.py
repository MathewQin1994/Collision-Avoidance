# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
from src.planner.Astar_jit import DeliberativePlanner
from numpy import pi,cos,sin
from src.map.staticmap import Map, generate_do_trajectory
from src.demo.simulation import simulation1
# from src.experiment.do_tra_predict import do_tra_predict
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import logging

colors = ['white', 'gold', 'orange', 'blue', 'green', 'purple']
bounds = [0,1,2,3,4,5,6]


cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

def do_tra_predict(s0,target_points):
    speed=s0[3]
    def lines(s0,s1):
        d=np.sqrt((s0[0]-s1[0])**2+(s0[1]-s1[1])**2)
        t=round(d/speed)-1
        yaw=np.arctan2(s1[1]-s0[1],s1[0]-s0[0])
        tra = np.zeros((int(t), 5))
        for i in range(tra.shape[0]):
            tra[i, :] = [s0[0] + i * speed* cos(yaw),s0[1] + i * speed * sin(yaw), yaw, speed, i]
        return tra
    do_tra=lines(s0,target_points[0])
    for j in range(len(target_points)-1):
        do_tra=np.vstack((do_tra,lines(target_points[j],target_points[j+1])))
    do_tra[:,-1]=list(range(do_tra.shape[0]))
    return do_tra

if __name__=="__main__":

    # 地图、起点、目标
    static_map = Map()
    static_map.load_map(np.loadtxt('../map/static_map6.txt', dtype=np.int8), resolution=1,offset=(-80,-35))
    s0 = tuple(np.array((2, 52, 0, 0, 0), dtype=np.float64))
    sG = tuple(np.array((43, -49, pi, 0.8, 0), dtype=np.float64))
    # s0 = tuple(np.array((35/2, 114/2, 0, 0.8, 0), dtype=np.float64))
    # sG = tuple(np.array((36, 46), dtype=np.float64))



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
    fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    fig.plot(sG[1], sG[0], "ob", markersize=5)
    # plt.show()
    # 本船参数和规划器
    resolution_time = 1
    resolution_pos = 1
    default_speed = 0.8
    primitive_file_path = '../primitive/control_primitives5.npy'
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

    do_s0['1'] = (48, -72, pi/2, 0.5, 0)
    do_goal['1'] = [(18, 20)]
    # do_s0['2'] = (24, -40, 0, 0.6, 0)
    # do_goal['2'] = [(5, 63)]


    for key in do_s0:
        do_tra_true[key] = do_tra_predict(do_s0[key], do_goal[key])

    simulation1(s0, sG, dp, fig, do_tra_true, predict_time=6)
    plt.show()