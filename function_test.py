import numpy as np
from numpy import pi,exp
import matplotlib.pyplot as plt
import time

def collision_pro_montecarlo(pos_usv,pos_ob,cov,radius,sample_num,plot_show=False):
    s=np.random.multivariate_normal(pos_ob,cov,sample_num)
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


def collision_pro_cal(pos_usv,pos_ob,cov,radius):
    stdx,stdy=cov[0][0],cov[1][1]
    p = 1 / (2 * pi * stdx * stdy) * exp(-1 / 2 * (((pos_usv[0]-pos_ob[0]) / stdx) ** 2 + ((pos_usv[1]-pos_ob[1]) / stdy) ** 2)) * pi*radius**2
    return p


def test_montecarlo():
    sample_nums=100*np.array(range(1,11))
    p=[]
    for sample_num in sample_nums:
        p.append(collision_pro_montecarlo(pos_usv,pos_ob,cov,radius,sample_num))
    fig = plt.figure().gca()
    fig.plot(sample_nums,p)
    plt.show()

def compare():
    start_time=time.time()
    for i in range(10000):
        p1 = collision_pro_montecarlo(pos_usv, pos_ob, cov, radius, 800, plot_show=False)
    print(time.time()-start_time)

    start_time=time.time()
    for i in range(10000):
        p2 = collision_pro_cal(pos_usv, pos_ob, cov, radius)
    print(time.time()-start_time)


def plot_test():
    import numpy as np
    from matplotlib.patches import Circle, Wedge, Polygon
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


if __name__=="__main__":
    # test()
    # std=5
    # pos_usv=(3,3)
    # pos_ob=(10,10)
    # cov=[[std**2,0],[0,std**2]]
    # radius=4
    plot_test()