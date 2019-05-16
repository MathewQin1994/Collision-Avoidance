import numpy as np
from math import sin,cos,pi
import matplotlib.pyplot as plt
from msgdev import PeriodTimer
import time
from PID import PIDcontroller

# 粘性阻力系数Cr=(Xuu,Xrr,Yv,Yr,Nv,Nr,Nrr)
# Cr=(-6.7,15.9,-29.5,11.8,-0.17,-2.74,-4.78)
# #质量与附加质量M=(mx,my,Izz)
# M=(33.3,58.0,6.1)
# #螺旋桨推力系数T=(Tnn,Tun)
# T=(0.01205,-0.0644)

dt=0.1

def acceleration(u,v,r,n1,n2):
    ax=(58.0*r*v-6.7*u**2+15.9*r**2+0.01205*
        (n1**2+n2**2)-0.0644*(u*(n1+n2)+0.45*r*(n1-n2)))/33.3
    ay=(-33.3*r*u-29.5*v+11.8*r)/58
    ar=(-0.17*v-2.74*r-4.78*r*abs(r)+0.45*
        (0.01205*(n1**2-n2**2)-0.0644*(u*(n1-n2)+0.45*r*(n1+n2))))/6.1
    return ax,ay,ar

def state_update(s,n1,n2):
    u, v, r, x, y, yaw=s
    ax,ay,ar=acceleration(u,v,r,n1,n2)
    u1=u+ax*dt+0.005*np.random.randn()
    v1=v+ay*dt+0.005*np.random.randn()
    r1=r+ar*dt+0.001*np.random.randn()
    x1=x+(u*cos(yaw)-v*sin(yaw))*dt
    y1=y+(u*sin(yaw)+v*cos(yaw))*dt
    yaw1=yaw+r*dt
    return (u1,v1,r1,x1,y1,yaw1)

def simulation(s0,n1s,n2s):
    length=len(n1s)
    s=s0
    t=PeriodTimer(dt)
    t.start()
    start_time=time.time()
    for i in range(length):
        with t:
            s=state_update(s,n1s[i],n2s[i])
            plt.plot(i,s[0],"ok")
            plt.pause(0.0001)
    print("actual time is {},supposed time is{}".format(time.time()-start_time,dt*length))
    plt.show()

def yaw_keeping(s0,target_yaw):
    s=s0
    ave_speed=600/60
    pid=PIDcontroller(800/60,3/60,10/60,dt)
    fig=plt.figure()
    a1=fig.add_subplot(2,2,1)
    a1.set_xlabel('t/s')
    a1.set_ylabel('yaw/rads')
    a2=fig.add_subplot(2,2,2)
    a2.set_xlabel('y')
    a2.set_ylabel('x')
    a2.axis("equal")
    a3=fig.add_subplot(2,2,3)
    a3.set_xlabel('t/s')
    a2.set_ylabel('u/m*s-1')
    start_time = time.time()
    t=PeriodTimer(dt)
    t.start()
    while True:
        with t:
            u=pid.update(target_yaw-s[5])
            print(pid.PTerm,pid.DTerm)
            n1=ave_speed+u/2
            n2=ave_speed-u/2
            s=state_update(s,n1,n2)
            a1.plot(t.i,s[5],"ok",markersize=2)
            a2.plot(s[4],s[3],"ob",markersize=2)
            a3.plot(t.i,s[0],'ok',markersize=2)
            plt.pause(0.0001)
        if t.i>1000:
            break
        # print("actual time is {},supposed time is{}".format(time.time() - start_time, dt * t.i))
    plt.show()

def speedkeeping(s0,target_speed):
    s=s0
    ave_speed=target_speed*19.56
    pid=PIDcontroller(800/60,3/60,10/60,dt)
    fig=plt.figure()
    a1=fig.add_subplot(2,2,1)
    a2=fig.add_subplot(2,2,2)
    a3=fig.add_subplot(2,2,3)
    a2.axis("equal")
    start_time = time.time()
    t=PeriodTimer(dt)
    t.start()
    while True:
        with t:
            u=pid.update(target_speed-s[0])
            print(pid.PTerm,pid.DTerm)
            n1=ave_speed+u/2
            n2=ave_speed+u/2
            s=state_update(s,n1,n2)
            a1.plot(t.i,s[5],"ok")
            a2.plot(s[4],s[3],"ob")
            a3.plot(t.i,s[0],'ok')
            plt.pause(0.0001)
        if t.i>1000:
            break
        # print("actual time is {},supposed time is{}".format(time.time() - start_time, dt * t.i))
    plt.show()

def control_action_primitives(s0,target_speed,target_yaw,plot=False):
    s=s0
    s_all=[s]
    yaw_control=PIDcontroller(800/60,3/60,10/60,dt)
    speed_control=PIDcontroller(800/60,3/60,10/60,dt)
    ave_speed = target_speed * 19.56
    if plot:
        fig=plt.figure()
        a1=fig.add_subplot(2,2,1)
        a1.set_xlabel('t/s')
        a1.set_ylabel('yaw/rads')
        a2=fig.add_subplot(2,2,2)
        a2.set_xlabel('y')
        a2.set_ylabel('x')
        a2.axis("equal")
        a3=fig.add_subplot(2,2,3)
        a3.set_xlabel('t/s')
        a2.set_ylabel('u/m*s-1')
    i=0
    while True:
        d_ave=speed_control.update(target_speed-s[0])
        diff=yaw_control.update(target_yaw-s[5])
        n1=ave_speed+d_ave+diff/2
        n2=ave_speed+d_ave-diff/2
        s=state_update(s,n1,n2)
        s_all.append(s)
        # if t.i%5==0:
        if plot:
            a1.plot(i*dt,s[5],"ok")
            a1.plot([i*dt,(i+1)*dt],[target_yaw,target_yaw],"-r")
            a2.plot(s[4],s[3],"ob")
            a3.plot(i*dt,s[0],'ok')
            a3.plot([i*dt,(i+1)*dt],[target_speed,target_speed],"-r")
            # plt.pause(0.0001)
        if (target_speed-s0[0])*(target_speed-s[0])<=0 and (target_yaw-s0[5])*(target_yaw-s[5])<=0:
            break
        i+=1
    if plot:
        plt.show()
    return s_all

if __name__=="__main__":
    s0=(1.2,0,0,0,0,0)
    speed_set=(0.0,0.4,0.8,1.2)
    yaw_set=(-pi/2,-pi/4,0.0,pi/4,pi/2)
    # n1s=[1000/60]*100
    # n2s=[1000/60]*100
    # simulation(s0,n1s,n2s)
    # yaw_keeping(s0,pi/4)
    # speedkeeping(s0,0.3)
    s_all=control_action_primitives(s0,1.2,pi/4,plot=True)