# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
import matplotlib.pyplot as plt
from src.control.PID import PIDcontroller
from src.control.tra_follow import control_action_primitives,state_update,yawRange

dt=0.1
yaw_control = PIDcontroller(800, 3, 10, dt)
speed_control = PIDcontroller(800, 3, 10, dt)


def zuotu(save=False):
    # time_set=np.array([10,5],dtype=np.int)
    u=0.8
    control_primitives=dict()
    action_time=10
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 12, -pi / 4,0, pi / 12, pi / 4], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(u,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 10
    yaw_set = np.array([-pi / 6, -pi / 3,pi / 6, pi / 3], dtype=np.float64)
    for yaw in yaw_set:
        key = (u, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    u=1.2
    action_time=10
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 12, -pi / 4, 0, pi / 12, pi / 4], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(u,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 10
    yaw_set = np.array([-pi / 6, -pi / 3, pi / 6, pi / 3], dtype=np.float64)
    for yaw in yaw_set:
        key = (u, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    return control_primitives

def zuotu1(s0,target_speed,target_yaw,action_time):

    yaw_control=PIDcontroller(800,3,10,dt)
    speed_control=PIDcontroller(3200,3,10,dt)
    propeller_speed = target_speed * 19.56*60
    d=(3/pi*abs(target_yaw)+2)*target_speed/0.8
    delta=10
    fig=plt.gca()
    fig.set_xlabel('y/m')
    fig.set_ylabel('x/m')
    fig.axis("equal")

    color=['r','g','b','y']
    ds=[d/2,0.75*d,1.05*d,1.5*d]
    for d,c in zip(ds,color):
        s = s0
        primitives_state = []
        i=1
        l=0
        fig.plot([0, (delta/2-d/2) * tan(target_yaw)], [d, delta/2+d/2], "--{}".format(c))
        while True:
            #s=(u,v,r,x,y,yaw)
            e=abs(cos(target_yaw))*(s[4]-(s[3]-d)*tan(target_yaw))
            alpha=np.arctan2(e,delta)

            d_pro=speed_control.update(target_speed-s[0])
            diff=yaw_control.update(yawRange(target_yaw-alpha-s[5]))
            n1 = propeller_speed + d_pro + diff * 480 / propeller_speed
            n2 = propeller_speed + d_pro - diff * 480 / propeller_speed
            if n1 > 1500:
                n1 = 1500
            elif n1 < -1500:
                n1 = -1500
            if n2 > 1500:
                n2 = 1500
            elif n2 < -1500:
                n2 = -1500
            # print(n1,n2)
            l +=s[0]*dt
            s=state_update(s,n1,n2)
            # print(e,alpha,n1,n2)
            primitives_state.append((s[3],s[4],s[5],s[0],i*dt,l))

            # fig.plot(s[4],s[3],"o{}".format(c),markersize=2)
            # plt.pause(0.0001)
            if i >= action_time/dt:
                break
            i+=1
        primitives_state=np.array(primitives_state)
        fig.plot(primitives_state[:,1],primitives_state[:,0],c)
    plt.show()
        # print("u0:{} u:{} yaw:{} distance:{} time:{}".format(s0[0],target_speed,target_yaw,np.sqrt(s[3]**2+s[4]**2),i*dt))

def zuotu2(save=False):
    # time_set=np.array([10,5],dtype=np.int)
    u=0.8
    control_primitives=dict()
    action_time=8
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 12, -pi / 4,0, pi / 12, pi / 4], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(u,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 8
    yaw_set = np.array([-pi / 6, -pi / 3,pi / 6, pi / 3], dtype=np.float64)
    for yaw in yaw_set:
        key = (u, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    yaw_set = np.array([0], dtype=np.float64)
    for yaw in yaw_set:
        key = (0, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),0,yaw,action_time,plot=False),dtype=np.float64)

    u=0
    action_time=8
    control_primitives[u]=dict()
    yaw_set = np.array([0], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(0.8,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),0.8,yaw,action_time,plot=False),dtype=np.float64)


    return control_primitives

if __name__=="__main__":
    # control_primitives=zuotu()
    # control_primitives_visual(control_primitives)
    # zuotu1(s0,0.8,pi/4,8)
    control_primitives=zuotu2()
