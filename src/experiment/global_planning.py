# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.planner.Astar_jit import DeliberativePlanner
from src.map.staticmap import Map
import time

dT=12
max_length=200
control_primitives = np.load('../primitive/control_primitives.npy',allow_pickle=True).item()
sg = tuple(np.array((41/2, 71/2, pi, 0.8, 0), dtype=np.float64))

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def generate_target_points(dev,control_primitives,n,fig=None):
    default_speed=0.8
    s=dev.sub_get('USV150.state')
    s=[s[3],s[4],s[5],s[0],time.time()]
    target_points=s.copy()
    for _ in range(n):
        current_yaw = s[2]
        current_pos = np.array(s[0:2])
        current_time = s[4]
        current_speed = round(s[3] / default_speed) * default_speed
        keys=list(control_primitives[current_speed].keys())
        key=keys[np.random.randint(0,len(keys))]
        ucd=control_primitives[current_speed][key]
        state_all=[]
        for i in range(ucd.shape[0]):
            state_all.append([current_pos[0] + ucd[i, 0] * cos(current_yaw) - ucd[i, 1] * sin(current_yaw),
                              current_pos[1] + ucd[i, 0] * sin(current_yaw) + ucd[i, 1] * cos(current_yaw),
                              yawRange(current_yaw + key[1] * pi / 180),
                              ucd[i, 3],
                              current_time + ucd[i, 4]])
        s = state_all[-1]
        target_points.extend(s)
        if fig:
            fig.plot(s[1],s[0],"ob", markersize=2)
            state_all=np.array(state_all)
            fig.plot(state_all[:,1],state_all[:,0],'b')
    # plt.show()
    return target_points

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
    dev.sub_connect('tcp://127.0.0.1:55007')
    dev.sub_connect('tcp://127.0.0.1:55009')
    dev.sub_connect('tcp://127.0.0.1:55001')  # receive rpm from joystick
    dev.sub_add_url('js.autoctrl',default_values=1)
    dev.pub_bind('tcp://0.0.0.0:55008')
    dev.sub_add_url('USV150.state', default_values=(0, 0, 0, 0, 0, 0))
    dev.sub_add_url('do_tra', default_values=[0]*(max_length*5*3))
    dev.sub_add_url('do_num')
    t = PeriodTimer(dT)

    #地图
    static_map = Map()
    static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), resolution=0.5)
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
    time.sleep(0.5)
    return dev,t,dp


if __name__=='__main__':
    try:
        dev,t,dp=initialize()
        target_points=[]
        t.start()
        while True:
            with t:
                autoctrl = dev.sub_get1('js.autoctrl')
                if autoctrl:
                    start_time = time.time()+dT
                    state = dev.sub_get('USV150.state')
                    if len(target_points)==0:
                        dev.pub_set('idx-length',[0,0])
                        dev.pub_set('target_points',[0]*(max_length*5))
                        s0 = (state[3], state[4], state[5], state[0], 0)
                    else:
                        dev.pub_set('idx-length',[t.i-1,target_points.shape[0]])
                        # print(target_points)
                        ta1=target_points.flatten().tolist()
                        ta1.extend([0] * (max_length * 5 - len(ta1)))
                        dev.pub_set('target_points',ta1)
                        s0=(tra[dT,0],tra[dT,1],tra[dT,2],tra[dT,3],0)
                        print(state[3:],tra[0,:3])

                    # a=get_virtual_do_tra(do_tra_true,start_time)
                    do_num = int(dev.sub_get1('do_num'))
                    if do_num > 0:
                        do_tra = np.array(dev.sub_get('do_tra')).reshape((3, max_length, 5))
                        do_tra = do_tra[:do_num,  int(start_time - do_tra[0, 0, -1]):,:]
                        dp.set_dynamic_obstacle(do_tra)
                    target_points = np.array(dp.start(s0, sg,tra_type='target_points'))
                    target_points[:, -1] = target_points[:, -1] + start_time
                    tra = target_points[0:1,:]
                    for i in range(target_points.shape[0] - 1):
                        tra = np.vstack((tra, compute_trajectory(target_points[i, :], target_points[i + 1, :])))

                    # length=np.random.randint(10,20)
                    # target_points = generate_target_points(dev,control_primitives, length)
                    # target_points.extend([0]*(max_length*5-len(target_points)))

                    # print('idx:{},length:{}'.format(t.i,length))
                    # print(len(target_points))
                else:
                    dev.pub_set('idx-length', [0, 0])
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set('idx-length', [0, 0])
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()

