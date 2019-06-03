import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Map:
    def add_static_obstacle(self,type,config):
        if type=="rectangle":
            #config=(y,x,dy,dx)
            x1=max(config[1],0)
            y1=max(config[0], 0)
            x2=min(config[1]+config[3],self.size[1])
            y2=min(config[0]+config[2], self.size[0])
            self.map[y1:y2,x1:x2]=1

    def load_map(self,map,resolution,offset=(0,0)):
        self.map=map
        self.resolution = resolution
        self.offset = offset
        self.size=map.shape

    def new_map(self,size,resolution,offset=(0,0)):
        #size=(size_y,size_x)
        self.size=size
        self.resolution=resolution
        self.offset=offset
        self.map=np.zeros(size,dtype=np.int8)


class DynamicObstacle:
    def __init__(self,radius,s0):
        #s0=(posx,posy,speed_angle,speed,x_var,y_var)
        self.radius=radius
        self.states=np.zeros((100,5),dtype=np.float32)
        self.states[0,:]=s0


def generate_do_trajectory(x0,y0,yaw,u,t):
    #tra=(x,y,yaw,stdx,stdy)
    tra=np.zeros((int(t),3))
    for i in range(tra.shape[0]):
        tra[i,:]=[x0+i*u*cos(yaw),y0+i*u*sin(yaw),yaw]
    return tra



if __name__=="__main__":
    map_size=(100,100)
    # rectangle_static_obstacles=((20,20,10,30),(50,20,30,10),(50,50,30,10))
    rectangle_static_obstacles = ((20, 50, 40, 10), (50, 20, 10, 30))
    map=Map(map_size)
    fig=plt.gca()
    fig.axis([0,map_size[0],0,map_size[1]])
    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle",config=ob)
        rect = patches.Rectangle((ob[0],ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)

    plt.show()


    # tra=generate_do_trajectory(50,100,-pi/2,1,100)
