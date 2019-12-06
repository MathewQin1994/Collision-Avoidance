# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
import matplotlib.pyplot as plt
# from msgdev import PeriodTimer
import time
from src.control.PID import PIDcontroller
from src.experiment.trajectory_follow import cal_target_yaw_t,cal_target_speed

dt=0.1
yaw_control = PIDcontroller(800, 3, 10, dt)
speed_control = PIDcontroller(800, 3, 10, dt)

def acceleration(u,v,r,n1,n2):
    ax=(58.0*r*v-6.7*u*abs(u)+15.9*r**2+0.01205*
        (n1*abs(n1)+n2*abs(n2))-0.0644*(u*(abs(n1)+abs(n2))+0.45*r*(abs(n1)-abs(n2))))/33.3
    ay=(-33.3*r*u-29.5*v+11.8*r)/58
    ar=(-0.17*v-2.74*r-4.78*r*abs(r)+0.45*
        (0.01205*(n1*abs(n1)-n2*abs(n2))-0.0644*(u*(abs(n1)-abs(n2))+0.45*r*(abs(n1)+abs(n2)))))/6.1
    return ax,ay,ar

def state_update(s,n1,n2):
    n1,n2=n1/60,n2/60
    u, v, r, x, y, yaw=s
    ax,ay,ar=acceleration(u,v,r,n1,n2)
    u1=u+ax*dt
    v1=v+ay*dt
    r1=r+ar*dt
    x1=x+(u*cos(yaw)-v*sin(yaw))*dt
    y1=y+(u*sin(yaw)+v*cos(yaw))*dt
    yaw1=yaw+r*dt
    return (u1,v1,r1,x1,y1,yaw1)

def state_update_noise(s,n1,n2):
    n1,n2=n1/60,n2/60
    u, v, r, x, y, yaw=s
    ax,ay,ar=acceleration(u,v,r,n1,n2)
    u1=u+ax*dt+0.005*np.random.randn()
    v1=v+ay*dt+0.005*np.random.randn()
    r1=r+ar*dt+0.001*np.random.randn()
    x1=x+(u*cos(yaw)-v*sin(yaw))*dt+0.01*np.random.randn()+0.005
    y1=y+(u*sin(yaw)+v*cos(yaw))*dt+0.01*np.random.randn()+0.005
    yaw1=yaw+r*dt+0.001*np.random.randn()
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
    propeller_speed = target_speed * 19.56*60
    d=(3/pi*abs(target_yaw)+2)*target_speed/0.8
    delta=10
    yaw_control = PIDcontroller(800, 3, 10, dt)
    speed_control = PIDcontroller(800, 3, 10, dt)
    if plot:
        fig=plt.figure()
        a1=fig.add_subplot(2,2,1)
        a1.set_xlabel('t/s')
        a1.set_ylabel('yaw/rads')
        a2=fig.add_subplot(2,2,2)
        a2.set_xlabel('y')
        a2.set_ylabel('x')
        a2.axis("equal")
        a2.plot([0,(delta-d)*tan(target_yaw)],[d,delta],"-r")
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
        n1=propeller_speed +d_pro+diff*480/propeller_speed
        n2=propeller_speed +d_pro-diff*480/propeller_speed
        if n1>1500:
            n1=1500
        elif n1<-1500:
            n1=-1500
        if n2>1500:
            n2=1500
        elif n2<-1500:
            n2=-1500
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
    action_time=8
    control_primitives[u]=dict()
    yaw_set = np.array([-pi / 3,-pi / 4, -pi / 6,-pi / 12, 0, pi / 12,pi / 6, pi / 4,pi / 3], dtype=np.float64)
    # yaw_set = np.array([-pi / 4, 0, pi / 4], dtype=np.float64)
    for yaw in yaw_set:
        key=(action_time,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    # action_time = 6
    # yaw_set = np.array([-pi/3,-pi / 6,pi / 6, pi / 3], dtype=np.float64)
    # for yaw in yaw_set:
    #     key = (action_time, np.int(np.round(yaw * 180 / pi)))
    #     control_primitives[u][key]=np.array(control_action_primitives((u,0,0,0,0,0),u,yaw,action_time,plot=False),dtype=np.float64)

    action_time = 8
    control_primitives[0.0] = dict()
    yaw_set = np.array([0], dtype=np.float64)
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
    fig=plt.figure().gca()
    fig.set_xlabel('y/m')
    fig.set_ylabel('x/m')
    fig.axis("equal")
    for i, k in enumerate(control_primitives.keys()):
        for key,c in control_primitives[k].items():
            fig.plot([0]+c[:,1].tolist(),[0]+c[:,0].tolist())
            fig.annotate("({},{},{},{})".format(k,key[0],key[1],c.shape[0]),(c[-1,1],c[-1,0]))
    plt.show()

def trajectory_following(s0,target_points,type=2,fig=None):
    s=(s0[0]+0.1,s0[1]+0.1,s0[2]+0.01,s0[3]+1,s0[4]+1,s0[5]+0.1)
    actual_points=[[s[3],s[4]]]
    record=[]
    delta=10
    i=1
    for p in range(1,len(target_points)):
        if type==2:
            old_target_x, old_target_y,_,_,old_target_t= target_points[p-1]
            target_x, target_y, target_yaw, target_speed, target_t = target_points[p]
            target_speed=cal_target_speed(old_target_x, old_target_y, old_target_t, target_x, target_y, target_t, target_speed, s[3], s[4])
        else:
            target_x, target_y, target_yaw, target_speed, target_t = target_points[p]
        propeller_speed = target_speed * 19.56*60

        if fig:
            fig.plot(target_y, target_x, "or", markersize=5)
        while i<target_t/dt:
            if type==0:
                target_yaw_t=target_yaw
            else:
                target_yaw_t = cal_target_yaw_t(target_x, target_y, target_yaw, s[3], s[4], delta)
            d_pro = speed_control.update(target_speed - s[0])
            diff = yaw_control.update(yawRange(target_yaw_t-s[5]))
            # print(target_yaw,target_yaw_t,s[5],diff,target_speed,s)
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
            #record=[x,y,target_yaw,yaw,target_speed,u,n1,n2]
            record.append([s[3],s[4],target_yaw_t, s[5], target_speed, s[0], n1, n2])
            s = state_update_noise(s, n1, n2)
            if fig:
                fig.plot(s[4], s[3], "ob", markersize=2)
                plt.pause(0.01)
            i+=1
        actual_points.append([s[3], s[4]])
    return np.array(actual_points),np.array(record)


def generate_target_points(s0,control_primitives,n,fig=None):
    default_speed=0.8
    s=[s0[3],s0[4],s0[5],s0[0],0]
    target_points=[s]
    state_all = []
    annotate=[]
    for num in range(n):
        current_yaw = s[2]
        current_pos = np.array(s[0:2])
        current_time = s[4]
        current_speed = round(s[3] / default_speed) * default_speed
        keys=list(control_primitives[current_speed].keys())
        np.random.seed(num+10)
        key=keys[np.random.randint(0,len(keys))]
        ucd=control_primitives[current_speed][key]
        for i in range(ucd.shape[0]):
            state_all.append([current_pos[0] + ucd[i, 0] * cos(current_yaw) - ucd[i, 1] * sin(current_yaw),
                              current_pos[1] + ucd[i, 0] * sin(current_yaw) + ucd[i, 1] * cos(current_yaw),
                              yawRange(current_yaw + ucd[i, 2]),
                              ucd[i, 3],
                              current_time + ucd[i, 4]])

        s=state_all[-1]
        target_points.append(s)
        annotate.append(("({},{},{},{})".format(current_speed, default_speed, key[1], ucd.shape[0])))
    target_points=np.array(target_points)
    state_all=np.array(state_all)
    if fig:
        fig.plot(target_points[:,1],target_points[:,0],"ob", markersize=3)
        fig.plot(state_all[:,1],state_all[:,0],'b--')
    return target_points,state_all,annotate

def zuotu(target_points,actual_points,state_all,record,annotate=None):
    fig=plt.figure()
    times=np.array(range(record.shape[0]))*0.1
    #位置时历
    fig1=fig.add_subplot(2,2,1)
    fig1.set_ylabel('N/m')
    fig1.set_xlabel('E/m')
    fig1.axis("equal")
    fig1.plot(target_points[:, 1], target_points[:, 0], "or", markersize=3,label="目标状态位置")
    fig1.plot(state_all[:, 1], state_all[:, 0], 'b--',label='规划轨迹')
    fig1.plot(record[:,1],record[:,0],'b',label='实际轨迹')
    fig1.plot(actual_points[:, 1], actual_points[:, 0], "ob", markersize=3, label="目标状态时刻的实际位置")
    for i,anno in enumerate(annotate):
        fig1.annotate(anno,(target_points[i+1,1]+2,target_points[i+1,0]))
    fig1.legend()

    #期望艏向和实际艏向
    fig2=fig.add_subplot(2,2,3)
    fig2.set_ylabel('surge speed/ms-1')
    fig2.set_xlabel('t/s')
    fig2.plot(times,record[:,4],label="瞬时期望纵向速度")
    fig2.plot(times,record[:,5], label="实际纵向速度")
    fig2.legend()

    fig3=fig.add_subplot(2,2,2)
    fig3.set_ylabel('yaw/rad')
    fig3.set_xlabel('t/s')
    fig3.plot(times,record[:,2],label="瞬时期望艏向角")
    fig3.plot(times,record[:,3], label="实际艏向角")
    fig3.legend()

    fig4=fig.add_subplot(2,2,4)
    fig4.set_ylabel('propeller speed/rpm')
    fig4.set_xlabel('t/s')
    fig4.plot(times,record[:,6],label="左桨转速")
    fig4.plot(times,record[:,7], label="右桨转速")
    fig4.legend()
    plt.show()

def compare():
    s0 = (0, 0, 0, 0, 0,0)
    control_primitives=np.load('../primitive/control_primitives.npy',allow_pickle=True).item()
    target_points,state_all,annotate=generate_target_points(s0,control_primitives,10)
    actual_points0,record0=trajectory_following(s0, target_points,type=0)
    actual_points1, record1 = trajectory_following(s0, target_points, type=1)
    actual_points2, record2 = trajectory_following(s0, target_points, type=2)
    tmp0=target_points[:,:2]-actual_points0
    dis0 = [np.inner(tmp0[i,:], tmp0[i,:]) for i in range(tmp0.shape[0])]
    tmp1=target_points[:,:2]-actual_points1
    dis1 = [np.inner(tmp1[i,:], tmp1[i,:]) for i in range(tmp1.shape[0])]
    tmp2=target_points[:,:2]-actual_points2
    dis2 = [np.inner(tmp2[i,:], tmp2[i,:]) for i in range(tmp2.shape[0])]
    fig1=plt.gca()
    fig1.set_ylabel('distance/m')
    fig1.set_xlabel('t/s')
    times=target_points[:,4]
    fig1.plot(times,dis0,label="阶跃信号控制器")
    fig1.plot(times,dis1, label="经典LOS循迹控制器")
    fig1.plot(times,dis2, label="改进LOS循迹控制器")
    fig1.legend()
    # zuotu(target_points,actual_points,state_all,record)

if __name__=="__main__":
    s0 = (0, 0, 0, 0, 0,0)
    # target_speed=0.8
    # target_yaw=pi/12
    # action_time=10
    # primitives_state=np.array(control_action_primitives((0,0,0,0,0,0), 0.8, 0, 8, plot=False))
    # control_primitives=get_all_control_primitives(save=True)
    # control_primitives_visual(control_primitives)
    # control_primitives_visual(control_primitives)


    # control_primitives=np.load('../primitive/control_primitives.npy',allow_pickle=True).item()
    # target_points,state_all,annotate=generate_target_points(s0,control_primitives,10)
    # actual_points,record=trajectory_following(s0, target_points,type=2)
    # zuotu(target_points,actual_points,state_all,record,annotate=annotate)

    compare()