import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Map:
    def __init__(self,size,offset=(0,0)):
        #size=(size_y,size_x)
        self.size=size
        self.offset=offset
        self.map=np.zeros(size,dtype=np.int8)

    def add_static_obstacle(self,type,config):
        if type=="rectangle":
            #config=(y,x,dy,dx)
            x1=max(config[1],0)
            y1=max(config[0], 0)
            x2=min(config[1]+config[3],self.size[1])
            y2=min(config[0]+config[2], self.size[0])
            self.map[y1:y2,x1:x2]=1



class DynamicObstacle:
    def __init__(self,radius,s0):
        #s0=(posx,posy,speed_angle,speed,x_var,y_var)
        self.radius=radius
        self.states=np.zeros((100,5),dtype=np.float32)
        self.states[0,:]=s0



if __name__=="__main__":
    map_size = (50, 50)
    # rectangle_static_obstacles=((20,20,10,30),(50,20,30,10),(50,50,30,10))
    rectangle_static_obstacles = ((20, 30, 10, 2), (30, 20, 2, 10))
    map=Map(map_size)
    fig=plt.gca()
    fig.axis([0,map_size[0],0,map_size[1]])
    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle",config=ob)
        rect = patches.Rectangle((ob[0],ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)
    plt.show()
