# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.planner.Astar_jit import DeliberativePlanner
from src.map.staticmap import Map
import time

max_length=200
dt=1
def get_virtual_do_tra(do_tra_true,start_time):
    do_tra=[]
    for key in do_tra_true:
        idx=int(start_time - do_tra_true[key][0, -1])
        if idx<do_tra_true[key].shape[0]:
            tra=do_tra_true[key][idx:,:]
        else:
            tra=do_tra_true[key][-1:,:]
        if tra.shape[0] > 200:
            tra = tra[:200, :]
        elif tra.shape[0] < 200:
            add = np.zeros((200 - tra.shape[0], 5))
            add[:, 0] = tra[-1, 0]
            add[:, 1] = tra[-1, 1]
            add[:, 2] = tra[-1, 2]
            add[:, 4] = np.linspace(tra[-1,-1]+1,tra[-1,-1]+200 - tra.shape[0],200 - tra.shape[0])
            tra = np.vstack((tra, add))
        do_tra.append(tra)
    do_tra=np.array(do_tra)
    return do_tra

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
    do_tra[:,-1]=np.linspace(time.time()+ 1, time.time() + do_tra.shape[0],do_tra.shape[0])
    return do_tra

def pub_do_tra(dev,do_tra):

    do1 = do_tra.flatten().tolist()
    do1.extend([0] * (max_length * (5*3) - len(do1)))
    dev.pub_set1('do_num', do_tra.shape[0])
    dev.pub_set('do_tra',do1)
    for i in range(do_tra.shape[0]):
        print('do1,x:{},y:{}'.format(do_tra[i,0,0],do_tra[i,0,1]))


# 他船参数和规划器
do_s0=dict()
do_dp = dict()
do_tra_true = dict()
do_goal=dict()

do_s0['1']=(71, 2, 1.57, 0.8, 0)
do_goal['1'] = [(87,22),(53/2, 121/2)]
do_s0['2']=(40/2, 114/2, 0, 0.8, 0)
do_goal['2'] = [(56,48),(76, 99)]
# do_s0['3'] = (159/2, 155/2, pi, 0.4, 0)
# do_goal['3'] = (99/2, 93/2)
# do_s0['4']=(75, 78, -pi/2, 0.4, 0)
# do_goal['4'] = (52, 49)

for key in do_s0:
    do_tra=do_tra_predict(do_s0['2'],do_goal['2'])
    do_tra_true[key] = do_tra_predict(do_s0[key],do_goal[key])




if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        dev.pub_bind('tcp://0.0.0.0:55009')
        # print(target_points)
        t = PeriodTimer(dt)
        t.start()
        while True:
            with t:
                start_time=time.time()
                do_tra = get_virtual_do_tra(do_tra_true, start_time)
                pub_do_tra(dev,do_tra)
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set1('do_num', 0)
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()