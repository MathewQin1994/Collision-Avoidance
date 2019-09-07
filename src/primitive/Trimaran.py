import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.pyplot as plt
# from msgdev import PeriodTimer
import time
from src.control.PID import PIDcontroller

# 粘性阻力系数Cr=(Xuu,Xrr,Yv,Yr,Nv,Nr,Nrr)
# Cr=(-6.7,15.9,-29.5,11.8,-0.17,-2.74,-4.78)
# #质量与附加质量M=(mx,my,Izz)
# M=(33.3,58.0,6.1)
# #螺旋桨推力系数T=(Tnn,Tun)
# T=(0.01205,-0.0644)

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
    # u1=u+ax*dt+0.005*np.random.randn()
    # v1=v+ay*dt+0.005*np.random.randn()
    # r1=r+ar*dt+0.001*np.random.randn()
    u1=u+ax*dt
    v1=v+ay*dt
    r1=r+ar*dt
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

def control_action_primitives(s0,target_speed,target_yaw,action_time,plot=False,STOP=True):
    s=s0
    primitives_state=[]
    yaw_control=PIDcontroller(800/60,3/60,10/60,dt)
    speed_control=PIDcontroller(3200/60,3/60,10/60,dt)
    propeller_speed = target_speed * 19.56
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
        a3.set_ylabel('u/m*s-1')
    i=1
    speed_stop=False
    yaw_stop=False
    l=0
    while True:
        d_pro=speed_control.update(target_speed-s[0])
        diff=yaw_control.update(target_yaw-s[5])
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

        if i%10==0:
            primitives_state.append((s[3],s[4],s[5],s[0],i*dt,l))
        if plot:
            a1.plot(i*dt,s[5],"ok",markersize=2)
            a1.plot([i*dt,(i+1)*dt],[target_yaw,target_yaw],"-r")
            a2.plot(s[4],s[3],"ob",markersize=2)
            a3.plot(i*dt,s[0],'ok',markersize=2)
            a3.plot([i*dt,(i+1)*dt],[target_speed,target_speed],"-r")
            plt.pause(0.0001)
        if STOP:
            if not speed_stop:
                if target_speed-s0[0]==0:
                    speed_stop = True
                elif (target_speed-s0[0])*(target_speed-0.2*abs(target_speed-s0[0])/(target_speed-s0[0])-s[0])<=0:
                    speed_stop=True
            if not yaw_stop:
                if (target_yaw-s0[5])*(target_yaw-s[5])<=0:
                    yaw_stop=True
            # if yaw_stop and speed_stop and i%10==0 and i>=action_time*10:
            if i >= action_time * 10:
                break
        i+=1
    if plot:
        plt.show()
    print("u0:{} u:{} yaw:{} distance:{} time:{}".format(s0[0],target_speed,target_yaw,np.sqrt(s[3]**2+s[4]**2),i*dt))
    return primitives_state

def get_all_control_primitives(save=True):
    # time_set=np.array([10,5],dtype=np.int)
    u=1.2
    control_primitives=dict()
    action_time=10
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 4, -pi / 12, 0, pi / 12, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(action_time,"{:.2f}".format(yaw))
        key=(action_time,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False,STOP=True),dtype=np.float64)

    action_time = 6
    yaw_set = np.array([-pi/3,-pi / 6, pi / 6, pi / 3], dtype=np.float64)
    for yaw in yaw_set:
        key=(action_time,"{:.2f}".format(yaw))
        key = (action_time, np.int(np.round(yaw * 180 / pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False,STOP=True),dtype=np.float64)

    action_time = 10
    control_primitives[0.0] = dict()
    yaw_set = np.array([-pi / 4, -pi / 6, -pi / 12, 0, pi / 12, pi / 6, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key = (action_time, "{:.2f}".format(yaw))
        key = (action_time, np.int(np.round(yaw * 180 / pi)))
        control_primitives[0.0][key] = np.array(
            control_action_primitives((0.0, 0, 0, 0, 0, 0), u, yaw, action_time, plot=False, STOP=True), dtype=np.float64)
    if save:
        np.save('control_primitives{}.npy'.format(u),control_primitives)
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

if __name__=="__main__":
    # s0=(0.8,0,0,0,0,0)
    # control_action_primitives(s0, 0.8, pi/3,6, plot=True, STOP=True)
    control_primitives=get_all_control_primitives(save=True)
    # control_primitives=np.load('control_primitives.npy').item()
    control_primitives_visual(control_primitives)
