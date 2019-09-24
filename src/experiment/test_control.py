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
speed_control = PIDcontroller(800, 3, 10, dt)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


def body2NE(posx,posy,yaw,dx,dy,dyaw):
    x1=posx+dx*cos(yaw)-dy*sin(yaw)
    y1=posy+dx*sin(yaw)+dy*cos(yaw)
    yaw1=yawRange(yaw+dyaw)
    return x1,y1,yaw1

def cal_target_yaw_t(x0,y0,target_yaw,xob,yob,delta):
    x1=sin(target_yaw)*cos(target_yaw)*(yob-y0)+cos(target_yaw)**2*xob+sin(target_yaw)**2*x0+delta*cos(target_yaw)
    y1=sin(target_yaw)*cos(target_yaw)*(xob-x0)+cos(target_yaw)**2*y0+sin(target_yaw)**2*yob+delta*sin(target_yaw)
    target_yaw_t=np.arctan2(y1-yob,x1-xob)
    return target_yaw_t

def test_control(dyaw,target_speed,dev):
    delta=10
    step=0
    for i in range(5):
        s = dev.sub_get('USV150.state')
        time.sleep(0.1)
    d = (3 / pi * abs(dyaw) + 2) * target_speed / 0.8
    target_x,target_y,target_yaw=body2NE(s[3],s[4],s[5],d,0,dyaw)
    # print(target_x,target_y,target_yaw)
    propeller_speed = target_speed * c_speed2motor*60
    t=PeriodTimer(dt)
    t.start()
    while True:
        with t:
            s_ob = list(dev.sub_get('USV150.state'))
            dr.write(s_ob+[time.time()])
            if step==0 and t.i>100:
                d = (3 / pi * abs(dyaw) + 2) * target_speed / 0.8
                target_x, target_y, target_yaw = body2NE(s_ob[3], s_ob[4], s_ob[5], d, 0, dyaw)
                step=1
            target_yaw_t = cal_target_yaw_t(target_x, target_y, target_yaw, s_ob[3], s_ob[4], delta)
            e = abs(cos(target_yaw) * (s_ob[4] - target_y - tan(target_yaw) * (s_ob[3] - target_x)))
            d_pro = speed_control.update(target_speed - s_ob[0])
            diff = yaw_control.update(yawRange(target_yaw_t - s_ob[5]))
            n1 = propeller_speed + d_pro + diff / 2
            n2 = propeller_speed + d_pro - diff / 2
            if n1 > 1500:
                n1 = 1500
            elif n1 < -1500:
                n1 = -1500
            if n2 > 1500:
                n2 = 1500
            elif n2 < -1500:
                n2 = -1500
            # print(n1,n2,s_ob)
            dev.pub_set1('pro.left.speed', -n1)
            dev.pub_set1('pro.right.speed', n2)
            print("e:{:.2f},t_yaw:{:.2f},t_yaw_t:{:.2f},yaw:{:.2f},left:{:.0f},right:{:.0f}".format(e,target_yaw,target_yaw_t,s_ob[5],n1,n2))

if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        if sys.argv[1] == 'simulation':
            dev.sub_connect('tcp://127.0.0.1:55007')
            dev.pub_bind('tcp://0.0.0.0:55002')
        elif sys.argv[1] == 'usv152':
            dev.sub_connect('tcp://192.168.1.152:55207')
            dev.pub_bind('tcp://0.0.0.0:55202')
        else:
            dev.sub_connect('tcp://192.168.1.150:55007')
            dev.pub_bind('tcp://0.0.0.0:55002')
        dev.sub_connect('tcp://127.0.0.1:55008')
        dev.sub_add_url('USV150.state',default_values=(0,0,0,0,0,0))
        dev.sub_add_url('idx-length',default_values=[0,0])
        dev.sub_add_url('target_points', default_values=[0]*(max_length*5))

        dyaw=sys.argv[2]
        filename = os.path.split(__file__)[-1].split(".")[0] + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))+'_'+dyaw+'.txt'
        filename='../data_record/{}'.format(filename)
        dr = DataRecord(filename)
        test_control(int(dyaw)/180*pi,0.8,dev)
    except (KeyboardInterrupt,Exception) as e:
        dev.pub_set1('pro.left.speed', 0)
        dev.pub_set1('pro.right.speed', 0)
        dr.close()
        time.sleep(0.5)
        dev.close()
        raise
    finally:
        dev.close()