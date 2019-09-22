# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
from src.control.PID import PIDcontroller
from src.tools.data_record import DataRecord
import time
import os
max_length=200
dt=0.2
c_speed2motor=19.56
yaw_control = PIDcontroller(800, 3, 10, dt)
speed_control = PIDcontroller(3200, 3, 10, dt)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def cal_target_yaw_t(x0,y0,target_yaw,xob,yob,delta):
    x1=sin(target_yaw)*cos(target_yaw)*(yob-y0)+cos(target_yaw)**2*xob+sin(target_yaw)**2*x0+delta*cos(target_yaw)
    y1=sin(target_yaw)*cos(target_yaw)*(xob-x0)+cos(target_yaw)**2*y0+sin(target_yaw)**2*yob+delta*sin(target_yaw)
    target_yaw_t=np.arctan2(y1-yob,x1-xob)
    return target_yaw_t

def trajectory_following(dev):
    delta=10
    p=0
    idx_old=-1
    old_target_x, old_target_y, old_target_t = 0.0,0.0,0.0
    t=PeriodTimer(dt)
    t.start()
    while True:
        idx,length=(int(item) for item in dev.sub_get('idx-length'))
        if length==0 or p==length-1:
            dev.pub_set1('pro.left.speed', 0)
            dev.pub_set1('pro.right.speed', 0)
            continue
        elif idx==idx_old:
            p+=1
        else:
            idx_old=idx
            target_points=np.array(dev.sub_get('target_points')).reshape(max_length,5)
            target_points=target_points[:length,:]
            p=0

        s_ob = dev.sub_get('USV150.state')
        target_x, target_y, target_yaw, target_speed, target_t = target_points[p]
        if old_target_t!=0.0:
            pre_distance=np.sqrt((old_target_x-target_x)**2+(old_target_y-target_y)**2)
            true_distance=np.sqrt((s_ob[3]-target_x)**2+(s_ob[4]-target_y)**2)
            target_speed=target_speed+(true_distance-pre_distance)/(target_t-old_target_t)
        old_target_x,old_target_y,old_target_t=target_x,target_y,target_t
        propeller_speed = target_speed * c_speed2motor*60
        # print('target speed',target_speed)
        while time.time()<target_t:
            # print(target_x, target_y, target_yaw, target_speed, target_t)
            with t:
                idx, length = (int(item) for item in dev.sub_get('idx-length'))
                if idx!=idx_old:
                    break
                s_ob = dev.sub_get('USV150.state')
                target_yaw_t=cal_target_yaw_t(target_x,target_y,target_yaw,s_ob[3],s_ob[4],delta)
                e=abs(cos(target_yaw)*(s_ob[4]-target_y-tan(target_yaw)*(s_ob[3]-target_x)))
                d_pro = speed_control.update(target_speed - s_ob[0])
                diff = yaw_control.update(yawRange(target_yaw_t-s_ob[5]))
                n1 = propeller_speed + d_pro + diff /2
                n2 = propeller_speed + d_pro - diff /2
                if n1 > 1500:
                    n1 = 1500
                elif n1 < -1500:
                    n1 = -1500
                if n2 > 1500:
                    n2 = 1500
                elif n2 < -1500:
                    n2 = -1500
                # print(n1,n2,s_ob)
                dev.pub_set1('pro.left.speed', n1)
                dev.pub_set1('pro.right.speed', n2)
                print("e:{:.2f},t_speed:{:.2f},t_yaw:{:.2f},t_yaw_t:{:.2f},yaw:{:.2f},left:{:.0f},right:{:.0f}".format(e,target_speed, target_yaw,target_yaw_t, s_ob[5],n1,n2))

if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        if sys.argv[1] == 'simulation':
            dev.sub_connect('tcp://127.0.0.1:55007')
        else:
            dev.sub_connect('tcp://192.168.1.150:55007')
        dev.sub_connect('tcp://127.0.0.1:55008')
        dev.pub_bind('tcp://0.0.0.0:55002')
        dev.sub_add_url('USV150.state',default_values=(0,0,0,0,0,0))
        dev.sub_add_url('idx-length',default_values=[0,0])
        dev.sub_add_url('target_points', default_values=[0]*(max_length*5))
        trajectory_following(dev)
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set1('pro.left.speed', 0)
        dev.pub_set1('pro.right.speed', 0)
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()

