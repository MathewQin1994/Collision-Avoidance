# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
from src.planner.Astar_jit import DeliberativePlanner
from numpy import pi
from src.map.staticmap import Map, generate_do_trajectory
from src.experiment.do_tra_predict import get_virtual_do_tra
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
colors = ['white', 'gold', 'orange', 'blue', 'green', 'purple']
bounds = [0,1,2,3,4,5,6]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
color_order=['g','r']*2


def do_predict(dp,s0,sG):
    s0=(s0[0],s0[1],s0[2],round(s0[3] / 0.4) * 0.4,s0[4])
    tra = np.array(dp.start(s0, sG))
    if tra.shape[0]>200:
        tra=tra[:200,:]
    elif tra.shape[0]<200:
        add=np.zeros((200-tra.shape[0],5))
        add[:, 0] = tra[-1, 0]
        add[:, 1] = tra[-1, 1]
        add[:, 2] = tra[-1, 2]
        add[:, 4] = list(range(tra.shape[0],200))
        tra=np.vstack((tra,add))
    return tra


def simulation(s0,sG,dp,do_dp,fig,do_tra_true=dict(),do_goal=dict(),predict_time=10):
    tra = None
    time_stamp = 0
    si=0
    while True:
        do_tra=[]
        for key in do_tra_true:
            if do_tra_true[key].shape[0] > time_stamp:
                do_tra.append(do_predict(do_dp[key], tuple(do_tra_true[key][time_stamp]), do_goal[key]))
            else:
                do_tra.append(do_predict(do_dp[key], tuple(do_tra_true[key][-1]), do_goal[key]))

        if len(do_tra_true.keys())>0:
            do_tra = np.array(do_tra)
            dp.set_dynamic_obstacle(do_tra)
        if tra is not None:
            if tra.shape[0] > predict_time:
                fig.plot(tra[:predict_time+1, 1], tra[:predict_time+1, 0], 'b')
                for key in do_tra_true:
                    fig.plot(do_tra_true[key][time_stamp:time_stamp + predict_time+1, 1],do_tra_true[key][time_stamp:time_stamp + predict_time+1, 0], 'r')
                s0 = (tra[predict_time, 0], tra[predict_time, 1], tra[predict_time, 2], round(tra[predict_time, 3] / 0.8) * 0.8, predict_time)
                if si==-1:
                    tra = np.array(dp.start(s0, sG,fig))
                else:
                    tra = np.array(dp.start(s0, sG))
            else:
                break
        else:
            tra = np.array(dp.start(s0, sG))
        time_stamp += predict_time

        try:
            for plot_line in plot_lines:
                fig.lines.remove(plot_line[0])
        except BaseException:
            pass
        plot_lines = []
        plot_lines.append(fig.plot(s0[1], s0[0], "ob", markersize=5))
        plot_lines.append(fig.plot(tra[:, 1], tra[:, 0], "--b"))
        for i,key in enumerate(do_tra_true):
            if do_tra_true[key].shape[0]>time_stamp:
                plot_lines.append(fig.plot(do_tra_true[key][time_stamp, 1], do_tra_true[key][time_stamp, 0], "or", markersize=5))
            plot_lines.append(fig.plot(do_tra[i,predict_time:, 1], do_tra[i,predict_time:, 0], "--r"))

        plt.pause(0.1)
        si+=1

def simulation1(s0,sG,dp,fig,do_tra_true=dict(),predict_time=6):
    tra = None
    time_stamp = 0
    si=0
    sj=-1
    while True:
        do_tra=get_virtual_do_tra(do_tra_true,time_stamp)

        if do_tra.shape[0]>0:
            dp.set_dynamic_obstacle(do_tra)
        if tra is not None:
            if tra.shape[0] > predict_time:
                if sj==0:
                    fig.plot(tra[:predict_time+1, 1], tra[:predict_time+1, 0], 'b',label='本船历史轨迹')
                else:
                    fig.plot(tra[:predict_time + 1, 1], tra[:predict_time + 1, 0], 'b')
                for i,key in enumerate(do_tra_true):
                    if i>1:
                        continue
                    if sj==0:
                        fig.plot(do_tra_true[key][time_stamp:time_stamp + predict_time+1, 1],do_tra_true[key][time_stamp:time_stamp + predict_time+1, 0], color_order[i],label="他船{}历史轨迹".format(key))
                    else:
                        fig.plot(do_tra_true[key][time_stamp:time_stamp + predict_time + 1, 1],
                                 do_tra_true[key][time_stamp:time_stamp + predict_time + 1, 0], color_order[i])
                s0 = (tra[predict_time, 0], tra[predict_time, 1], tra[predict_time, 2], round(tra[predict_time, 3] / 0.8) * 0.8, 0)
                if si==-7:
                    tra = np.array(dp.start(s0, sG,fig))
                else:
                    tra = np.array(dp.start(s0, sG))
            else:
                break
        else:
            tra = np.array(dp.start(s0, sG))
        time_stamp += predict_time

        try:
            for plot_line in plot_lines:
                fig.lines.remove(plot_line[0])
        except BaseException:
            pass
        plot_lines = []
        plot_lines.append(fig.plot(s0[1], s0[0], 'b*',markersize=8,label='本船当前位置'))
        plot_lines.append(fig.plot(tra[:, 1], tra[:, 0], 'b--',label='本船规划轨迹'))
        for i,key in enumerate(do_tra_true):
            if i<2:
                if do_tra_true[key].shape[0]>time_stamp:
                    plot_lines.append(fig.plot(do_tra_true[key][time_stamp, 1], do_tra_true[key][time_stamp, 0], color_order[i]+'*', markersize=8,label="他船{}当前位置".format(key)))
                plot_lines.append(fig.plot(do_tra[i,predict_time:, 1], do_tra[i,predict_time:, 0], color_order[i]+'--',label="他船{}预测轨迹".format(key)))
            # else:
            #     plot_lines.append(
            #         fig.plot(do_tra[i, predict_time:, 1], do_tra[i, predict_time:, 0], color_order[i] + '--'))
            # else:
            #     if do_tra_true[key].shape[0]>time_stamp:
            #         plot_lines.append(fig.plot(do_tra_true[key][time_stamp, 1], do_tra_true[key][time_stamp, 0], '*r', markersize=8))
            #     plot_lines.append(fig.plot(do_tra[i,predict_time:, 1], do_tra[i,predict_time:, 0], 'r--'))
        fig.legend(loc='lower right')
        a = fig.text(120, 88, 'T={}s'.format(time_stamp))
        plt.pause(0.1)
        a.set_visible(False)
        si+=1
        sj+=1

if __name__=="__main__":
    # 参数
    resolution_time = 1
    resolution_pos = 1
    default_speed = 0.8
    primitive_file_path = '../primitive/control_primitives.npy'
    e = 1.2
    predict_time = 10

    # 地图、起点、目标
    map_size = (100, 100)
    static_map = Map()
    static_map.new_map(map_size, resolution=1, offset=(0, 0))
    fig = plt.gca()
    fig.axis([0 + static_map.offset[1], map_size[0] + static_map.offset[1], 0 + static_map.offset[0],
              map_size[1] + static_map.offset[0]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0 = tuple(np.array((10, 50, 0, 0.8, 10), dtype=np.float64))
    sG = tuple(np.array((95, 50, pi, 0.8, 0), dtype=np.float64))
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
    do_tra_true['1'] = generate_do_trajectory(95, 50, pi, 0.8, 200)
    # do_tra_true['2'] = generate_do_trajectory(85, 85, -pi * 3 / 4, 0.8, 200)
    do_goal['1']=(0, 50)
    # do_goal['2']=(0,0)

    #规划器
    dp = DeliberativePlanner(
        static_map,
        resolution_pos,
        resolution_time,
        default_speed,
        primitive_file_path,
        e)
    do_dp=dict()
    for key in do_tra_true:
        do_dp[key]=DeliberativePlanner(
            static_map,
            resolution_pos,
            resolution_time,
            default_speed,
            primitive_file_path,
            e)

    simulation(s0,sG,dp,do_dp,fig,do_tra_true,do_goal)
