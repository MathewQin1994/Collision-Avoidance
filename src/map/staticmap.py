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
            self.map[y1:y2+1,x1:x2+1]=1
        if type=='circle':
            #config=(y,x,r)
            y,x,r=config
            y=y-self.offset[0]
            x=x-self.offset[1]
            yrange=(max(y-r,0),min(y+r,self.map.shape[0]))
            for i in range(yrange[0],yrange[1]):
                d=int(round(np.sqrt(r**2-(y-i)**2)))
                xrange=(max(x-d,0),min(x+d,self.map.shape[1]))
                self.map[i,xrange[0]:xrange[1]]=1

    def load_map(self,map,resolution,offset=(0,0)):
        self.map=map
        self.resolution = resolution
        self.offset = offset
        self.size=map.shape

    def new_map(self,size,resolution=1,offset=(0,0)):
        #size=(size_y,size_x)
        self.size=size
        self.resolution=resolution
        self.offset=offset
        self.map=np.zeros(size,dtype=np.int8)

    def expand(self,width):
        map=self.map.copy()
        width=int(width/self.resolution)
        for i in range(width,map.shape[0]-width):
            for j in range(width,map.shape[1]-width):
                if map[i,j]==1:
                    self.map[i-width:i+width+1,j-width:j+width+1]=1

class DynamicObstacle:
    def __init__(self,radius,s0):
        #s0=(posx,posy,speed_angle,speed,x_var,y_var)
        self.radius=radius
        self.states=np.zeros((100,5),dtype=np.float32)
        self.states[0,:]=s0


def generate_do_trajectory(x0,y0,yaw,u,t):
    tra=np.zeros((int(t),5))
    for i in range(tra.shape[0]):
        tra[i,:]=[x0+i*u*cos(yaw),y0+i*u*sin(yaw),yaw,u,i]
    return tra



if __name__=="__main__":
    map_size=(100,100)
    # rectangle_static_obstacles=((20,20,10,30),(50,20,30,10),(50,50,30,10))
    # rectangle_static_obstacles = ((20, 50, 40, 10), (50, 20, 10, 30))
    # rectangle_static_obstacles=((90,150,110,50),(250,150,100,50),(150,250,100,100))
    rectangle_static_obstacles=()
    circle_static_obstacles=((40,60,10),)
    map=Map()
    map.new_map(map_size,1,offset=(0,0))

    fig=plt.gca()
    fig.axis([0,map_size[0],0,map_size[1]])
    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle",config=ob)
        rect = patches.Rectangle((ob[0],ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)
    for ob in circle_static_obstacles:
        map.add_static_obstacle(type="circle",config=ob)
    # map.expand(2)
    plt.imshow(map.map)
    plt.show()


    # tra=generate_do_trajectory(50,100,-pi/2,1,100)
