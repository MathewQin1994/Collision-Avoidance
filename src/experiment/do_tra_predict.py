# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi
from src.tools.msgdev import PeriodTimer,MsgDevice
import time

max_length=200
dt=1

def extend(tra):
    if tra.shape[0]==0:
        return tra
    elif tra.shape[0] > 200:
        tra = tra[:200, :]
    elif tra.shape[0] < 200:
        add = np.zeros((200 - tra.shape[0], 5))
        add[:, 0] = tra[-1, 0]
        add[:, 1] = tra[-1, 1]
        add[:, 2] = tra[-1, 2]
        add[:, 4] = np.linspace(tra[-1, -1] + 1, tra[-1, -1] + 200 - tra.shape[0], 200 - tra.shape[0])
        tra = np.vstack((tra, add))

    return tra

def get_virtual_do_tra(do_tra_true,start_time):
    do_tra=[]
    for key in do_tra_true:
        idx=int(start_time - do_tra_true[key][0, -1])
        if idx<do_tra_true[key].shape[0]:
            tra=do_tra_true[key][idx:,:]
        else:
            tra=do_tra_true[key][-1:,:]
        tra=extend(tra)
        do_tra.append(tra)
    do_tra=np.array(do_tra)
    return do_tra

def do_tra_predict(s0,target_points):
    speed=s0[3]
    def lines(s0,s1):
        d=np.sqrt((s0[0]-s1[0])**2+(s0[1]-s1[1])**2)
        t=round(d/speed)-1
        yaw=np.arctan2(s1[1]-s0[1],s1[0]-s0[0])
        tra = np.zeros((max(int(t),0), 5))
        for i in range(tra.shape[0]):
            tra[i, :] = [s0[0] + i * speed* cos(yaw),s0[1] + i * speed * sin(yaw), yaw, speed, i]
        return tra
    do_tra=lines(s0,target_points[0])
    for j in range(len(target_points)-1):
        do_tra=np.vstack((do_tra,lines(target_points[j],target_points[j+1])))
    do_tra[:,-1]=np.linspace(time.time()+ 1, time.time() + do_tra.shape[0],do_tra.shape[0])
    return do_tra

def pub_do_tra(dev,do_tra):
    do1 = do_tra.flatten().tolist()
    do1.extend([0] * (max_length * (5*3) - len(do1)))
    if do_tra.shape[1]==0:
        dev.pub_set1('do_num', 0)
        dev.pub_set('do_tra', do1)
    else:
        dev.pub_set1('do_num', do_tra.shape[0])
        dev.pub_set('do_tra',do1)


def generate_do_tra_true():
# 他船参数和规划器
    do_s0=dict()
    do_tra_true = dict()
    do_goal=dict()
    case=sys.argv[1]
    if case=='case1':
        do_s0['1']=(71, 2, 1.57, 0.8, 0)
        do_goal['1'] = [(87,22),(53/2, 121/2)]
        do_s0['2']=(40/2, 114/2, 0, 0.8, 0)
        do_goal['2'] = [(56,48),(76, 99)]
    elif case=='case2':
        do_s0['1']=(45, 45, 0.86, 0.8, 0)
        do_goal['1'] = [(100, 117)]
    elif case=='case3':
        do_s0['1'] = (17, 51, pi, 0.6, 0)
        do_goal['1'] = [(-28, 42)]
        do_s0['2'] = (24, -40, 0, 0.6, 0)
        do_goal['2'] = [(5, 63)]
    else:
        raise Exception
    for key in do_s0:
        do_tra_true[key] = do_tra_predict(do_s0[key],do_goal[key])
    return do_tra_true

def virtual_do_tra_predict(dev,t):
    do_tra_true = None
    while True:
        with t:
            autoctrl = dev.sub_get1('js.autoctrl')
            # print(autoctrl)
            if autoctrl:
                if do_tra_true is None:
                    do_tra_true = generate_do_tra_true()
                start_time = time.time()
                do_tra = get_virtual_do_tra(do_tra_true, start_time)
                pub_do_tra(dev, do_tra)

def real_do_tra_predict(dev,t):
    if sys.argv[1] == 'simulation':
        dev.sub_connect('tcp://127.0.0.1:55207')
        dev.sub_connect('tcp://127.0.0.2:55202')
    else:
        dev.sub_connect('tcp://192.168.1.152:55207')
        dev.sub_connect('tcp://192.168.1.152:55202')
    dev.sub_add_url('USV150.state', default_values=(0, 0, 0, 0, 0, 0))
    dev.sub_add_url('target.point', default_values=(0, 0))
    while True:
        with t:
            s_ob = dev.sub_get('USV150.state')
            target_points=[dev.sub_get('target.point')]
            s0=(s_ob[3],s_ob[4],s_ob[5],0.5,0)
            print(s_ob,target_points)
            do_tra=do_tra_predict(s0,target_points)
            do_tra=extend(do_tra)
            do_tra=do_tra.reshape((1,-1,5))
            pub_do_tra(dev, do_tra)



if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        dev.sub_connect('tcp://127.0.0.1:55001')  # receive rpm from joystick
        dev.sub_add_url('js.autoctrl',default_values=0)
        dev.pub_bind('tcp://0.0.0.0:55009')

        t = PeriodTimer(dt)
        t.start()
        if sys.argv[1] in {'simulation','real'}:
            real_do_tra_predict(dev, t)
        else:
            virtual_do_tra_predict(dev, t)
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set1('do_num', 0)
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()