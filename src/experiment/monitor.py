# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.map.staticmap import Map
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from collections import deque
import time

max_length=200
control_primitives = np.load('../primitive/control_primitives.npy').item()

x_q=deque()
y_q=deque()
idx_old=-1
def update_state(dev,fig,plot_lines):
    s_ob = dev.sub_get('USV150.state')
    x_q.append(s_ob[3])
    y_q.append(s_ob[4])
    if len(x_q)>60:
        x_q.popleft()
        y_q.popleft()
    # global plot_lines
    while len(plot_lines)>0:
        fig.lines.remove(plot_lines.pop()[0])
    plot_lines.append(fig.plot(y_q,x_q,'b'))


def update_planning(dev,fig,plot_lines):
    global idx_old
    idx, length = (int(item) for item in dev.sub_get('idx-length'))
    if length == 0:
        while len(plot_lines) > 0:
            fig.lines.remove(plot_lines.pop()[0])
    elif idx!=idx_old:

        idx_old = idx
        target_points = np.array(dev.sub_get('target_points')).reshape(max_length, 5)
        # print(target_points[:length,:])
        tra=np.zeros((0,5))
        for i in range(length-1):
            tra=np.vstack((tra,compute_trajectory(target_points[i,:],target_points[i+1,:])))
        while len(plot_lines) > 0:
            fig.lines.remove(plot_lines.pop()[0])
        plot_lines.append(fig.plot(tra[:, 1], tra[:, 0], '--b'))



def compute_trajectory(s_state, s1_state):
    # ucd=self.control_primitives[s_state[3]][(int(s1_state[4]-s_state[4]),"{:.2f}".format(yawRange(s1_state[2]-s_state[2])))]
    ucd = control_primitives[round(s_state[3] / 0.8) * 0.8][(int(
        s1_state[4] - s_state[4]), np.int(np.round(180 / pi * yawRange(s1_state[2] - s_state[2]))))]
    yaw = s_state[2]
    state_all = []
    for i in range(ucd.shape[0]):
        state_all.append([s_state[0] + ucd[i, 0] * cos(yaw) - ucd[i, 1] * sin(yaw),
                          s_state[1] + ucd[i, 0] * sin(yaw) + ucd[i, 1] * cos(yaw),
                          yawRange(yaw + ucd[i, 2]),
                          ucd[i, 3],
                          s_state[4] + ucd[i, 4]])
    return np.array(state_all)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

if __name__=='__main__':
    try:
        # 地图、起点、目标
        static_map = Map()
        static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), 1)
        s0 = tuple(np.array((174, 201, 0.86 - pi, 0.8, 10), dtype=np.float64))
        # s0 = tuple(np.array((190, 1, 1.57, 0.8, 0), dtype=np.float64))
        sG = tuple(np.array((41, 71, pi, 0.8, 0), dtype=np.float64))
        # sG = tuple(np.array((55, 180, pi, 0.8, 0), dtype=np.float64))

        fig = plt.gca()
        extend = [
            static_map.offset[0],
            static_map.size[0] *
            static_map.resolution +
            static_map.offset[0] - 1,
            static_map.offset[1],
            static_map.size[1] *
            static_map.resolution +
            static_map.offset[1] - 1]
        mapplot = static_map.map.copy()
        static_map.expand(2)
        for i in range(mapplot.shape[0]):
            mapplot[i, :] = mapplot[i, :][::-1]
        fig.imshow(mapplot.T, extent=extend)
        fig.set_xlabel('E/m')
        fig.set_ylabel('N/m')

        dev=MsgDevice()
        dev.open()
        dev.sub_connect('tcp://127.0.0.1:55007')
        dev.sub_connect('tcp://127.0.0.1:55008')
        dev.sub_add_url('USV150.state', default_values=(0, 0, 0, 0, 0, 0))
        dev.sub_add_url('idx-length', default_values=[0, 0])
        dev.sub_add_url('target_points', default_values=[0] * (max_length * 5))
        time.sleep(0.5)
        t=PeriodTimer(1)
        t.start()
        plot_lines_state=[]
        plot_lines_tra=[]
        while True:
            with t:
                dev.sub_get('target_points')
                update_state(dev,fig,plot_lines_state)
                update_planning(dev, fig, plot_lines_tra)
                plt.pause(0.1)

    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        raise
    finally:
        dev.close()