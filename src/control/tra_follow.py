# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
import matplotlib.pyplot as plt
# from msgdev import PeriodTimer
import time
from src.control.PID import PIDcontroller
# from src.primitive.Trimaran import state_update,dt

dt=0.1

def acceleration(u,v,r,n1,n2):
    ax=(58.0*r*v-6.7*u*abs(u)+15.9*r**2+0.01205*
        (n1*abs(n1)+n2*abs(n2))-0.0644*(u*(abs(n1)+abs(n2))+0.45*r*(abs(n1)-abs(n2))))/33.3
    ay=(-33.3*r*u-29.5*v+11.8*r)/58
    ar=(-0.17*v-2.74*r-4.78*r*abs(r)+0.45*
        (0.01205*(n1*abs(n1)-n2*abs(n2))-0.0644*(u*(abs(n1)-abs(n2))+0.45*r*(abs(n1)+abs(n2)))))/6.1
    return ax,ay,ar

def state_update(s,n1,n2):
    u, v, r, x, y, yaw=s
    ax,ay,ar=acceleration(u,v,r,n1,n2)
    # u1=u+ax*dt+0.01*np.random.randn()
    # v1=v+ay*dt+0.01*np.random.randn()
    # r1=r+ar*dt+0.005*np.random.randn()
    u1=u+ax*dt
    v1=v+ay*dt
    r1=r+ar*dt
    # x1=x+(u*cos(yaw)-v*sin(yaw))*dt+0.01*np.random.randn()
    # y1=y+(u*sin(yaw)+v*cos(yaw))*dt+0.01*np.random.randn()
    # yaw1=yaw+r*dt+0.01*np.random.randn()
    x1=x+(u*cos(yaw)-v*sin(yaw))*dt
    y1=y+(u*sin(yaw)+v*cos(yaw))*dt
    yaw1=yaw+r*dt
    return (u1,v1,r1,x1,y1,yaw1)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def control_action_primitives(s0,target_speed,target_yaw,action_time,plot=False):
    s=s0
    primitives_state=[]
    yaw_control=PIDcontroller(800/60,3/60,10/60,dt)
    speed_control=PIDcontroller(3200/60,3/60,10/60,dt)
    propeller_speed = target_speed * 19.56
    d=3/pi*abs(target_yaw)+2
    delta=10
    if plot:
        fig=plt.figure()
        a1=fig.add_subplot(2,2,1)
        a1.set_xlabel('t/s')
        a1.set_ylabel('yaw/rads')
        a2=fig.add_subplot(2,2,2)
        a2.set_xlabel('y')
        a2.set_ylabel('x')
        a2.axis("equal")
        a2.plot([0,(target_speed*action_time-d)*tan(target_yaw)],[d,target_speed*action_time],"-r")
        a3=fig.add_subplot(2,2,3)
        a3.set_xlabel('t/s')
        a3.set_ylabel('u/m*s-1')
    i=1
    l=0
    while True:
        #s=(u,v,r,x,y,yaw)
        e=abs(cos(target_yaw))*(s[4]-(s[3]-d)*tan(target_yaw))
        alpha=np.arctan2(e,delta)

        d_pro=speed_control.update(target_speed-s[0])
        diff=yaw_control.update(yawRange(target_yaw-alpha-s[5]))
        n1=propeller_speed +d_pro+diff*8/propeller_speed
        n2=propeller_speed +d_pro-diff*8/propeller_speed
        if n1>25:
            n1=25
        elif n1<-25:
            n1=-25
        if n2>25:
            n2=25
        elif n2<-25:
            n2=-25
        # print(n1,n2)
        l +=s[0]*dt
        s=state_update(s,n1,n2)
        # print(e,alpha,n1,n2)
        if i%10==0:
            primitives_state.append((s[3],s[4],s[5],s[0],i*dt,l))
        if plot:
            a1.plot(i*dt,s[5],"ok",markersize=2)
            a1.plot([i*dt,(i+1)*dt],[target_yaw,target_yaw],"-r")
            a2.plot(s[4],s[3],"ob",markersize=2)
            a3.plot(i*dt,s[0],'ok',markersize=2)
            a3.plot([i*dt,(i+1)*dt],[target_speed,target_speed],"-r")
            plt.pause(0.0001)
        if i >= action_time/dt:
            break
        i+=1
    if plot:
        plt.show()
    print("u0:{} u:{} yaw:{} distance:{} time:{}".format(s0[0],target_speed,target_yaw,np.sqrt(s[3]**2+s[4]**2),i*dt))
    return primitives_state

def get_all_control_primitives(save=True):
    # time_set=np.array([10,5],dtype=np.int)
    u=0.8
    control_primitives=dict()
    action_time=6
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 4, -pi / 12, 0, pi / 12, pi / 4], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(action_time,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 6
    yaw_set = np.array([-pi/3,-pi / 6,pi / 6, pi / 3], dtype=np.float64)
    for yaw in yaw_set:
        key = (action_time, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 6
    control_primitives[0.0] = dict()
    yaw_set = np.array([-pi / 4, -pi / 6, -pi / 12, 0, pi / 12, pi / 6, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key = (action_time, np.int(np.round(yaw * 180 / pi)))
        control_primitives[0.0][key] = np.array(
            control_action_primitives((0.0, 0, 0, 0, 0, 0), u, yaw, action_time, plot=False), dtype=np.float64)
    if save:
        np.save('control_primitives.npy'.format(u),control_primitives)
    return control_primitives

def control_primitives_visual(control_primitives):
    fig_num=len(control_primitives)
    x_num=2
    y_num=int(ceil(fig_num/x_num))
    fig=plt.figure()
    a=[]
    for i, k in enumerate(control_primitives.keys()):
        a.append(fig.add_subplot(y_num,x_num,i+1))
        a[i].set_xlabel('y/m')
        a[i].set_ylabel('x/m')
        a[i].set_title(str(k))
        for key,c in control_primitives[k].items():
            a[i].plot([0]+c[:,1].tolist(),[0]+c[:,0].tolist())
            a[i].annotate(str(key),(c[-1,1],c[-1,0]))
    plt.show()

def trajectory_following(s0,target_points,fig=None):

    s=s0
    s=(s0[0]+0.1,s0[1]+0.1,s0[2]+0.01,s0[3]+1,s0[4]+1,s0[5]+0.1)
    yaw_control=PIDcontroller(800/60,3/60,10/60,dt)
    speed_control=PIDcontroller(3200/60,3/60,10/60,dt)

    # d=3/pi*target_yaw+2
    delta=10
    i=1
    # l=0
    p=0
    while p<len(target_points):
        target_x, target_y, target_yaw, target_speed, target_t = target_points[p]
        propeller_speed = target_speed * 19.56
        # while np.sqrt((s[3]-target_x)**2+(s[4]-target_y)**2)>0.1:
        while i<target_t*10:
            s_ob = (
            s[0] + 0.005 * np.random.randn(), s[1] + 0.005 * np.random.randn(), s[2] + 0.001 * np.random.randn(),
            s[3] + 0.05 * np.random.randn(), s[4] + 0.05 * np.random.randn(), s[5] + 0.01 * np.random.randn())
            beta=np.arctan2(target_y-s_ob[4],target_x-s_ob[3])
            if yawRange(target_yaw-beta)>0:
                coe=1
            else:
                coe=-1
            e=coe*abs(cos(target_yaw)*(s_ob[4]-target_y-tan(target_yaw)*(s_ob[3]-target_x)))
            alpha = np.arctan2(e, delta)
            print(target_yaw,alpha,s_ob[5],e)
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
            # print(n1,n2)
            # l += s[0] * dt
            s = state_update(s, n1, n2)
            if fig:
                fig.plot(s[4], s[3], "ob", markersize=2)
                plt.pause(0.01)
            i+=1
        p+=1
        print(p)

def generate_target_points(s0,control_primitives,n,fig=None):
    default_speed=0.8
    s=s0
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



if __name__=="__main__":
    s0 = (0, 0, 0, 0, 0, 0)
    # target_speed=0.8
    # target_yaw=pi/12
    # action_time=10
    # control_action_primitives(s0, target_speed, target_yaw, action_time, plot=True)

    # control_primitives=get_all_control_primitives(save=True)
    # # control_primitives=np.load('control_primitives.npy').item()
    # control_primitives_visual(control_primitives)


    fig = plt.gca()
    fig.axis("equal")
    control_primitives=np.load('../primitive/control_primitives.npy',allow_pickle=True).item()
    target_points=generate_target_points(s0,control_primitives,10,fig)
    trajectory_following(s0, target_points, fig)