# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.planner.Astar_jit import DeliberativePlanner
from src.map.staticmap import Map
import time

dT=6
max_length=200
control_primitives = np.load('../primitive/control_primitives.npy',allow_pickle=True).item()
sg= tuple(np.array((41 / 2, 71 / 2, pi, 0.8, 0), dtype=np.float64))

def choose_case():
    global sg
    static_map = Map()
    case=sys.argv[2]
    if case=='case1':
        sg = tuple(np.array((41 / 2, 71 / 2, pi, 0.8, 0), dtype=np.float64))
        static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), resolution=0.5)
    elif case=='case2':
        sg = tuple(np.array((74 / 2, 150 / 2, pi, 0.8, 0), dtype=np.float64))
        static_map.load_map(np.loadtxt('../map/static_map2.txt', dtype=np.int8), resolution=0.5)
    elif case=='case3':
        sg = tuple(np.array((48, -9, pi, 0.8, 0), dtype=np.float64))
        static_map.load_map(np.loadtxt('../map/static_map3.txt', dtype=np.int8), resolution=1, offset=(-80,-35))
    else:
        raise Exception
    return static_map

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

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

def initialize():
    #通信和计时器
    dev = MsgDevice()
    dev.open()
    if sys.argv[1] == 'simulation':
        dev.sub_connect('tcp://127.0.0.1:55007')
    elif sys.argv[1] == 'usv152':
        dev.sub_connect('tcp://192.168.1.152:55207')
    else:
        dev.sub_connect('tcp://192.168.1.150:55007')
    dev.sub_connect('tcp://127.0.0.1:55009')
    dev.sub_connect('tcp://127.0.0.1:55001')
    dev.sub_add_url('js.autoctrl',default_values=0)
    dev.pub_bind('tcp://0.0.0.0:55008')
    dev.sub_add_url('USV150.state', default_values=(0, 0, 0, 0, 0, 0))
    dev.sub_add_url('do_tra', default_values=[0]*(max_length*5*3))
    dev.sub_add_url('do_num')
    t = PeriodTimer(dT)
    for _ in range(5):
        s_ob = dev.sub_get('USV150.state')
        time.sleep(0.1)
    #地图
    static_map=choose_case()
    static_map.expand(1)

    #轨迹规划器
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
    do_tra=np.ones((1,50,5))
    dp.set_dynamic_obstacle(do_tra)
    s0 = (s_ob[3], s_ob[4], s_ob[5], s_ob[0], 0)
    dp.start(s0,sg)
    return dev,t,dp

def first_run(dp,dev):
    start_time = time.time() + 2*dT
    s_ob = dev.sub_get('USV150.state')
    s0 = (s_ob[3], s_ob[4], s_ob[5], s_ob[0], 0)
    for i in range(20):
        do_num = int(dev.sub_get1('do_num'))
        time.sleep(0.1)
    print("do_num:{}".format(do_num))
    if do_num > 0:
        do_tra = np.array(dev.sub_get('do_tra')).reshape((3, max_length, 5))
        do_tra = do_tra[:do_num, int(start_time - do_tra[0, 0, -1]):, :]
        dp.set_dynamic_obstacle(do_tra)
    target_points = np.array(dp.start(s0, sg, tra_type='target_points'))
    target_points[:, -1] = target_points[:, -1] + start_time
    tra = target_points[0:1, :]
    for i in range(target_points.shape[0] - 1):
        tra = np.vstack((tra, compute_trajectory(target_points[i, :], target_points[i + 1, :])))
    # if start_time>time.time():
    delay=start_time-time.time()
    print('delay',delay)
    time.sleep(delay)
    print('first run done')
    return target_points,tra

def plan(dev,t,dp):
    target_points, tra = first_run(dp, dev)
    t.start()
    while True:
        with t:
            start_time = time.time() + dT
            s_ob = dev.sub_get('USV150.state')
            dev.pub_set('idx-length', [t.i - 1, target_points.shape[0]])
            # print(target_points)
            # print(tra[7,:])
            ta1 = target_points.flatten().tolist()
            ta1.extend([0] * (max_length * 5 - len(ta1)))
            dev.pub_set('target_points', ta1)
            s0 = (tra[dT, 0], tra[dT, 1], tra[dT, 2], tra[dT, 3], 0)
            print("t_x:{:.2f},t_y:{:.2f},t_yaw:{:.2f},x:{:.2f},y:{:.2f},yaw:{:.2f}".format(
                tra[0,0],tra[0,1],tra[0,2],s_ob[3], s_ob[4], s_ob[5]))

            # a=get_virtual_do_tra(do_tra_true,start_time)
            do_num = int(dev.sub_get1('do_num'))
            if do_num > 0:
                do_tra = np.array(dev.sub_get('do_tra')).reshape((3, max_length, 5))
                do_tra = do_tra[:do_num, int(start_time - do_tra[0, 0, -1]):, :]
                dp.set_dynamic_obstacle(do_tra)
            target_points = np.array(dp.start(s0, sg, tra_type='target_points'))
            target_points[:, -1] = target_points[:, -1] + start_time
            tra = target_points[0:1, :]
            for i in range(target_points.shape[0] - 1):
                tra = np.vstack((tra, compute_trajectory(target_points[i, :], target_points[i + 1, :])))

if __name__=='__main__':
    try:
        dev,t,dp=initialize()
        print('initialize done')
        while True:
            autoctrl = dev.sub_get1('js.autoctrl')
            if autoctrl:
                break
            time.sleep(0.1)
        plan(dev,t,dp)
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set('idx-length', [0, 0])
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()

