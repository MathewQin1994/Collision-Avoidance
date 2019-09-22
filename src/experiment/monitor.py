# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.map.staticmap import Map
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import logging
from collections import deque
from src.tools.data_record import DataRecord
import time

dt=0.5
colors = ['white', 'gold', 'orange', 'blue', 'green', 'purple']
bounds = [0,1,2,3,4,5,6]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

max_length=200
control_primitives = np.load('../primitive/control_primitives.npy',allow_pickle=True).item()

x_q=deque()
y_q=deque()
idx_old=-1
def update_state(dev,fig,plot_lines):
    s_ob = list(dev.sub_get('USV150.state'))
    print("u:{:.2f},v:{:.2f},r:{:.2f},x:{:.2f},y:{:.2f},yaw:{:.2f},".format(*s_ob))
    x_q.append(s_ob[3])
    y_q.append(s_ob[4])
    if len(x_q)>60/dt:
        x_q.popleft()
        y_q.popleft()
    # global plot_lines
    while len(plot_lines)>0:
        fig.lines.remove(plot_lines.pop()[0])
    plot_lines.append(fig.plot(y_q,x_q,'b'))
    plot_lines.append(fig.plot(s_ob[4], s_ob[3], 'b*'))
    dr_state.write(s_ob+[time.time()])


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
        # tra=np.zeros((0,5))
        # for i in range(length-1):
        #     tra=np.vstack((tra,compute_trajectory(target_points[i,:],target_points[i+1,:])))
        while len(plot_lines) > 0:
            fig.lines.remove(plot_lines.pop()[0])
        # plot_lines.append(fig.plot(tra[:, 1], tra[:, 0], '--b'))
        for i in range(length):
            plot_lines.append(fig.plot(target_points[i,1], target_points[i,0], 'bo',markersize=2))
        dr_point.write(['shape',length,5])
        dr_point.write(target_points[:length,:])
        return 1
    return 0



def update_do_pre(dev,fig,plot_lines):
    do_num = int(dev.sub_get1('do_num'))
    if do_num > 0:
        do_tra = np.array(dev.sub_get('do_tra')).reshape((3, max_length, 5))
        do_tra = do_tra[:do_num, int(time.time() - do_tra[0, 0, -1]):, :]
        if do_tra.shape[1]>0 and update==1:
            dr_do.write(['shape',do_tra.shape[0],do_tra.shape[1],do_tra.shape[2]])
            dr_do.write(do_tra)

    while len(plot_lines)>0:
        fig.lines.remove(plot_lines.pop()[0])
    for i in range(do_num):
        plot_lines.append(fig.plot(do_tra[i,:,1],do_tra[i,:,0],'r--'))
        plot_lines.append(fig.plot(do_tra[i, 0, 1], do_tra[i, 0, 0], 'or',markersize=5))



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

def choose_case():
    static_map = Map()
    case=sys.argv[2]
    if case=='case1':
        static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), resolution=0.5)
    elif case=='case2':
        static_map.load_map(np.loadtxt('../map/static_map2.txt', dtype=np.int8), resolution=0.5)
    elif case=='case3':
        static_map.load_map(np.loadtxt('../map/static_map3.txt', dtype=np.int8), resolution=1, offset=(-63, -54))
    elif case=='case0':
        static_map.new_map(size=(100,100),offset=(-63, -54))
    else:
        raise Exception
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
    for i in range(mapplot.shape[0]):
        mapplot[i, :] = mapplot[i, :][::-1]
    fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    fig.axis("equal")
    return fig

def data_save():
    file_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    dir_path = '../data_record/global_planning/'
    state_file=dir_path+file_time+'_'+'state.txt'
    dr_state=DataRecord(state_file)
    do_file=dir_path+file_time+'_'+'do.txt'
    dr_do=DataRecord(do_file)
    target_point_file=dir_path+file_time+'_'+'point.txt'
    dr_point=DataRecord(target_point_file)
    return dr_state,dr_do,dr_point


if __name__=='__main__':
    try:
        # 地图、起点、目标
        fig=choose_case()
        dev=MsgDevice()
        dev.open()
        if sys.argv[1] == 'simulation':
            dev.sub_connect('tcp://127.0.0.1:55007')
        else:
            dev.sub_connect('tcp://192.168.1.150:55007')
        dev.sub_connect('tcp://127.0.0.1:55008')
        dev.sub_connect('tcp://127.0.0.1:55009')
        dev.sub_add_url('USV150.state', default_values=(0, 0, 0, 0, 0, 0))
        dev.sub_add_url('idx-length', default_values=[0, 0])
        dev.sub_add_url('target_points', default_values=[0] * (max_length * 5))
        dev.sub_add_url('do_tra', default_values=[0] * (max_length * 5 * 3))
        dev.sub_add_url('do_num')
        time.sleep(0.5)
        t=PeriodTimer(dt)
        t.start()
        plot_lines_state=[]
        plot_lines_tra=[]
        plot_lines_do_tra = []
        dr_state, dr_do, dr_point=data_save()
        while True:
            with t:
                update_state(dev,fig,plot_lines_state)
                update=update_planning(dev, fig, plot_lines_tra)
                update_do_pre(dev,fig,plot_lines_do_tra)
                plt.pause(0.1)

    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        dr_state.close()
        dr_do.close()
        dr_point.close()
        raise
    finally:
        dev.close()
