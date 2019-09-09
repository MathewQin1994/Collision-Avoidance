# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice
import time
from src.control.PID import PIDcontroller
import time

dt=0.1
yaw_control = PIDcontroller(800 / 60, 3 / 60, 10 / 60, dt)
speed_control = PIDcontroller(3200 / 60, 3 / 60, 10 / 60, dt)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def trajectory_following(dev,target_points):
    delta=10
    p=0
    t=PeriodTimer(dt)
    t.start()
    while p<len(target_points):
        target_x, target_y, target_yaw, target_speed, target_t = target_points[p]
        propeller_speed = target_speed * 19.56
        # while np.sqrt((s[3]-target_x)**2+(s[4]-target_y)**2)>0.1:
        while time.time()<target_t:
            with t:
                s_ob = dev.sub_get('USV150.state')
                beta=np.arctan2(target_y-s_ob[4],target_x-s_ob[3])
                if yawRange(target_yaw-beta)>0:
                    coe=1
                else:
                    coe=-1
                e=coe*abs(cos(target_yaw)*(s_ob[4]-target_y-tan(target_yaw)*(s_ob[3]-target_x)))
                alpha = np.arctan2(e, delta)
                # print(target_yaw,alpha,s_ob[5],e)
                d_pro = speed_control.update(target_speed - s_ob[0])
                diff = yaw_control.update(yawRange(target_yaw - alpha - s_ob[5]))
                n1 = propeller_speed + d_pro + diff * 8 / propeller_speed
                n2 = propeller_speed + d_pro - diff * 8 / propeller_speed
                if n1 > 25:
                    n1 = 25
                elif n1 < -25:
                    n1 = -25
                if n2 > 25:
                    n2 = 25
                elif n2 < -25:
                    n2 = -25
                print(n1,n2,s_ob)
                dev.pub_set1('pro.left.speed', n1)
                dev.pub_set1('pro.right.speed', n2)
        p+=1
        print(p)

def generate_target_points(control_primitives,n,fig=None):
    default_speed=0.8
    s=dev.sub_get('USV150.state')
    s=(s[3],s[4],s[5],s[0],time.time())
    target_points=[]
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
                              yawRange(current_yaw + ucd[i, 2]),
                              ucd[i, 3],
                              current_time + ucd[i, 4]])
        s = tuple(state_all[-1])
        target_points.append(s)
        if fig:
            fig.plot(s[1],s[0],"ob", markersize=2)
            state_all=np.array(state_all)
            fig.plot(state_all[:,1],state_all[:,0],'b')
    # plt.show()
    return target_points

if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        dev.sub_connect('tcp://127.0.0.1:55007')
        dev.pub_bind('tcp://0.0.0.0:55002')
        dev.sub_add_url('USV150.state',default_values=(0,0,0,0,0,0))
        control_primitives=np.load('../primitive/control_primitives.npy').item()
        target_points=generate_target_points(control_primitives,10)
        # print(target_points)
        trajectory_following(dev, target_points)
    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        raise
    else:
        dev.close()
    finally:
        dev.close()