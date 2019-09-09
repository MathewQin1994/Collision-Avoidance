# -*- coding:UTF-8 -*-
import sys
sys.path.append("../..")
import numpy as np
from numpy import sin,cos,pi,ceil,tan
from src.tools.msgdev import PeriodTimer,MsgDevice

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
    u1=u+ax*dt+0.01*np.random.randn()
    v1=v+ay*dt+0.01*np.random.randn()
    r1=r+ar*dt+0.005*np.random.randn()
    # u1=u+ax*dt
    # v1=v+ay*dt
    # r1=r+ar*dt
    x1=x+(u*cos(yaw)-v*sin(yaw))*dt+0.01*np.random.randn()
    y1=y+(u*sin(yaw)+v*cos(yaw))*dt+0.01*np.random.randn()
    yaw1=yawRange(yaw+r*dt+0.01*np.random.randn())
    return (u1,v1,r1,x1,y1,yaw1)

def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


if __name__=='__main__':
    try:
        dev=MsgDevice()
        dev.open()
        dev.sub_connect('tcp://127.0.0.1:55002')
        dev.pub_bind('tcp://0.0.0.0:55007')
        dev.sub_add_url('pro.left.speed')
        dev.sub_add_url('pro.right.speed')
        t=PeriodTimer(dt)
        t.start()
        s=(0,0,0,0,0,0)
        while True:
            with t:
                n1 = dev.sub_get1('pro.left.speed')
                n2 = dev.sub_get1('pro.right.speed')
                s=state_update(s,n1,n2)
                s_ob = (
                s[0] + 0.005 * np.random.randn(), s[1] + 0.005 * np.random.randn(), s[2] + 0.001 * np.random.randn(),
                s[3] + 0.05 * np.random.randn(), s[4] + 0.05 * np.random.randn(), s[5] + 0.01 * np.random.randn())
                dev.pub_set('USV150.state',s_ob)
                print(n1,n2,s_ob)
    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        raise
    else:
        dev.close()
    finally:
        pass