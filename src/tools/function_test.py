import numpy as np
from numpy import pi,exp,cos,sin
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
# from Astar import yawRange
import time
from numba import jit,prange
from numpy.linalg import cholesky
M=0
TCPA_MIN=15
DCPA_MIN=10
COLREGS_COST=1.0
a=np.array([np.array([[1,2],[2,3]]),np.array([[10,2],[2,3]])])

# @jit(nopython=True)
def collision_pro_montecarlo(pos_usv,pos_ob,sigma,radius,sample_num,plot_show=False):
    # s=np.random.multivariate_normal(pos_ob,cov,sample_num)
    s=np.vstack((np.random.normal(pos_ob[0],sigma[0],sample_num),np.random.normal(pos_ob[1],sigma[1],sample_num))).T
    distance=(s[:,0]-pos_usv[0])**2+(s[:,1]-pos_usv[1])**2
    p=np.sum(distance<radius**2)/sample_num
    if plot_show:
        fig = plt.figure().gca()
        fig.plot(s[:, 1], s[:, 0], 'ob', markersize=1)
        fig.plot(pos_usv[1], pos_usv[0], 'or', markersize=5)
        theta = np.linspace(0, 2 * np.pi, 800)
        y, x = np.cos(theta) * radius + pos_usv[0], np.sin(theta) * radius + pos_usv[1]
        fig.plot(y, x, "--r")
        plt.show()
    return p


# @jit(nopython=True)
# def collision_pro_cal(d,sigma,r):
#     x=np.arange(d-r,d+r,r/5)
#     return np.sum(1 / (2 * pi * sigma**2) * exp(-1 / 2 * x**2/sigma**2)*2*np.arccos((x**2+d**2-r**2)/2/x/d)*x*r/5)




# @jit(nopython=True)
def collision_pro_cal(d,sigma2,r):
    if d==0:
        step=r/10
        x1=np.arange(0,r,step)
        return np.sum(1 / (2 * pi * sigma2) * exp(-1 / 2 * x1 ** 2 / sigma2)*2*pi*x1*step)
    elif d>=r:
        step=(2*r)/10
        x=np.arange(d+r-0.1,d-r,-step)
        return np.sum(np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d)*2*x*step / (2 * pi * sigma2) * exp(-1 / 2 * x ** 2 / sigma2))
    else:
        step = (2*d - 0.1) / 10
        step1=(r-d)/10
        x=np.arange(d+r-0.05,r-d,-step)
        x1=np.arange(0,r-d,step1)
        b=(x ** 2 + d ** 2 - r ** 2) / 2 / x / d
        a=np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d)
        s=np.sum(np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d)*2*x*step / (2 * pi * sigma2) * exp(-1 / 2 * x ** 2 / sigma2))
        s1=np.sum(1 / (2 * pi * sigma2) * exp(-1 / 2 * x1 ** 2 / sigma2)*2*pi*x1*step1)
        return s+s1


def test_montecarlo():
    sample_nums=10000*np.array(range(1,11))
    p=[]
    for sample_num in sample_nums:
        p.append(collision_pro_montecarlo(pos_usv,pos_ob,sigma,radius,sample_num))
    fig = plt.figure().gca()
    fig.plot(sample_nums,p)
    plt.show()

@jit(nopython=True)
def compare(n):
    # collision_pro_montecarlo(pos_usv, pos_ob, sigma, radius, 800, plot_show=False)
    # d = np.sqrt(np.dot(pos_usv - pos_ob, pos_usv - pos_ob))
    # collision_pro_cal(d, std**2, radius)

    # start_time=time.time()
    # a=np.zeros(n)
    # for i in range(n):
    #     a[i] = collision_pro_montecarlo(pos_usv, pos_ob, sigma, radius, 800, plot_show=False)
    # print(time.time()-start_time)

    # start_time = time.time()
    a=np.zeros(n)
    for i in range(n):
        d = np.sqrt(np.dot(pos_usv - pos_ob, pos_usv - pos_ob))
        a[i] = collision_pro_cal(d, std**2, radius)
    # print(time.time()-start_time)


def plot_test():
    import numpy as np

    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, ax = plt.subplots()

    resolution = 50  # the number of vertices
    N = 3
    x = np.random.rand(N)
    y = np.random.rand(N)
    radii = 0.1 * np.random.rand(N)
    patches = []
    for x1, y1, r in zip(x, y, radii):
        circle = Circle((x1, y1), r)
        patches.append(circle)

    x = np.random.rand(N)
    y = np.random.rand(N)
    radii = 0.1 * np.random.rand(N)
    theta1 = 360.0 * np.random.rand(N)
    theta2 = 360.0 * np.random.rand(N)
    for x1, y1, r, t1, t2 in zip(x, y, radii, theta1, theta2):
        wedge = Wedge((x1, y1), r, t1, t2)
        patches.append(wedge)

    # Some limiting conditions on Wedge
    patches += [
        Wedge((.3, .7), .1, 0, 360),  # Full circle
        Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
        Wedge((.8, .3), .2, 0, 45),  # Full sector
        Wedge((.8, .3), .2, 45, 90, width=0.10),  # Ring sector
    ]

    for i in range(N):
        polygon = Polygon(np.random.rand(N, 2), True)
        patches.append(polygon)

    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax)

    plt.show()


def plot_ship(fig,x,y,yaw,l,b):
    x_p=l/2*cos(yaw)+x
    y_p=l/2*sin(yaw)+y
    theta=-(yaw*180/pi+90)
    dtheta=np.arcsin(b/2/l)*180/pi
    wedge=Wedge((y_p,x_p),l,theta-dtheta,theta+dtheta)
    fig.add_patch(wedge)

@jit(nopython=True)
def get_cpa(s1,s2):
    x1,y1,yaw1,u1,_=s1
    x2,y2,yaw2,u2,_=s2
    dv=np.array([u1*cos(yaw1)-u2*cos(yaw2),u1*sin(yaw1)-u2*sin(yaw2)])
    dpos=np.array([x1-x2,y1-y2])
    tcpa=-np.dot(dv,dpos)/np.dot(dv,dv)
    dpos1=dpos+tcpa*dv
    dcpa=np.sqrt(np.dot(dpos1,dpos1))
    return tcpa,dcpa

def test_cpa():
    s1=(1.0,1.0,0.79,0.8,10)
    s_info1=(1.2,1.2)
    s2=(1.0,17.0,-0.79,1.0,10)
    s_info2=(2.0,1.0)
    ax=plt.gca()
    ax.axis([0, 20, 0, 20])
    print(get_cpa(s1,s2))
    print(colrges_encounter_type(s1,s2))


    plot_ship(ax,s1[0],s1[1],s1[2],s_info1[0],s_info1[1])
    plot_ship(ax, s2[0], s2[1], s2[2], s_info2[0], s_info2[1])
    plt.show()

def colrges_encounter_type(s1,s2):
    x1,y1,yaw1,u1,_=s1
    x2,y2,yaw2,u2,_=s2
    alpha_b=yawRange(np.arctan2(y1-y2,x1-x2)-yaw2)
    alpha_h=yawRange(yaw1-yaw2)
    if abs(alpha_b)<=pi/12 and abs(alpha_h)>=11*pi/12:
        encounter_type="head on"
    elif alpha_b>pi/12 and alpha_b<3*pi/4 and alpha_h>-11*pi/12 and alpha_h<-pi/4:
        encounter_type="cross from left"
    elif alpha_b>-3*pi/4 and alpha_b<-pi/12 and alpha_h>pi/4 and alpha_h<11*pi/12:
        encounter_type="cross from right"
    elif abs(alpha_b)>=3*pi/4 and abs(alpha_h)<=pi/4:
        encounter_type="take over"
    else:
        encounter_type=None
    return encounter_type

def colrges_cost(s1,s2,encounter_type):
    x1,y1,yaw1,u1,_=s1
    x2,y2,yaw2,u2,_=s2
    alpha_b=yawRange(np.arctan2(y1-y2,x1-x2)-yaw2)
    if encounter_type=="head on" and alpha_b>-pi/8 and alpha_b<pi/2:
        cost=COLREGS_COST
    elif encounter_type=="cross from right" and alpha_b>-pi/4 and alpha_b<0.0:
        cost=COLREGS_COST
    else:
        cost=0.0
    return cost

def time_test():
    start_time=time.time()
    normal_dis(10000)
    print(time.time()-start_time)

    start_time=time.time()
    normal_dis(10000)
    print(time.time()-start_time)

@jit(nopython=True)
def forxunhuan(n):
    mu = np.array([[1, 5]])
    sigma = np.array([[1, 0.5], [1.5, 3]])
    for i in range(n):
        a=np.random.normal(mu, sigma, 500)
    return a

@jit(nopython=True)
def normal_dis(n):
    a=np.zeros((3,3))
    s=0
    for i,value in zip(a[:,0],a[:,1]):
        s +=np.arctan2(value,i)
    return s


@jit(nopython=True)
def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

@jit(nopython=True)
def colrges_cost(s_usv,s_ob,encounter_type):
    alpha_b=yawRange(np.arctan2(s_usv[1]-s_ob[1],s_usv[0]-s_ob[0])-s_ob[2])
    distance=np.sqrt(np.dot(s_usv[0:2]-s_ob[0:2],s_usv[0:2]-s_ob[0:2]))
    if encounter_type==4 and alpha_b>-pi/24 and alpha_b<pi/2 and distance<30:
        return 1.0
    elif encounter_type==3 and alpha_b>-pi/4 and alpha_b<pi/8 and distance<30:
        return 1.0
    else:
        return 0.0

if __name__=="__main__":
    # test_cpa()
    std=200
    pos_usv=np.array((0.0,0.0))
    pos_ob=np.array((0.0,1001.0))
    sigma=(std,std)
    radius=1000
    N=2400
    n=1900000
    test=(1.8237647235110979, 7.5, 4)

    # p1=collision_pro_montecarlo(pos_usv, pos_ob, sigma, radius, 8000, plot_show=True)
    # d = np.sqrt(np.inner(pos_usv - pos_ob, pos_usv - pos_ob))
    p2=collision_pro_cal(*test)
    # print(p1,p2)
    # start_time=time.time()
    # compare(100)
    # print(time.time()-start_time)
    #
    # start_time=time.time()
    # compare(n)
    # print(time.time()-start_time)
    # ts=np.arange(300)
    # plt.figure(2)
    # plt.plot(ts,np.exp(-0.01*ts))
    # stds=range(1,20)
    # p=[]
    # for std in stds:
    #     p.append(collision_pro_cal(d,std**2,radius))
    # plt.plot(stds,p)
    # f(np.array([1,2,3]),5,4,7)
    # start_time=time.time()
    # a=compare(n)
    # print(time.time()-start_time)
    # #
    # start_time=time.time()
    # for i in range(N):
    #     a=compare(n)
    # print(time.time()-start_time)
    # s=normal_dis(3)

    # start_time = time.time()
    # a=np.zeros(n)
    # for i in prange(n):
    #     a[i] = collision_pro_montecarlo(pos_usv, pos_ob, sigma, radius, 800, plot_show=False)
    # print(time.time() - start_time)
    #
    # start_time = time.time()
    # a=np.zeros(n)
    # for i in prange(n):
    #     a[i] = collision_pro_montecarlo(pos_usv, pos_ob, sigma, radius, 800, plot_show=False)
    # print(time.time() - start_time)


    # plot_test()
    # ax=plt.gca()
    # ax.axis([0, 20, 0, 20])
    # plot_ship(ax,5,5,0.79,2,1)
    # plt.show()
    # test_cpa()
    # time_test()
    # normal_dis()