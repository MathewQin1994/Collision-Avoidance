from numpy import sin,cos,pi
from map import Map,DynamicObstacle
import numpy as np
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt


YAW={'0.79':pi/4,'0.52':pi/6,'0.26':pi/12,'-0.79':-pi/4,'-0.52':-pi/6,'-0.26':-pi/12,'0.00':0}
map_size = (100, 100)
# rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
# rectangle_static_obstacles = ((20, 30, 12, 2), (30, 20, 2, 12))
map=Map(map_size)
for ob in rectangle_static_obstacles:
    map.add_static_obstacle(type="rectangle",config=ob)


fig=plt.gca()
fig.axis([0,map_size[0],0,map_size[1]])
fig.set_xlabel('E/m')
fig.set_ylabel('N/m')
for ob in rectangle_static_obstacles:
    map.add_static_obstacle(type="rectangle",config=ob)
    rect = patches.Rectangle((ob[0],ob[1]), ob[2], ob[3], color='y')
    fig.add_patch(rect)


class Node():
    def __init__(self,state,key,g=None,cost=None,ps=None,father=None):
        #state=(N,E,yaw,u,t)
        self.state=state
        self.key=key
        self.cost=cost
        self.ps=ps
        self.g=g
        self.father=father



class OpenList:
    def __init__(self):
        self._heap=[None]
        self.N=0
        self._idx=dict()

    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self._heap[1:])

    def swim(self,k):
        while k>1 and self._heap[k//2].cost>self._heap[k].cost:
            self._heap[k],self._heap[k//2]=self._heap[k//2],self._heap[k]
            self._idx[self._heap[k].key]=k
            self._idx[self._heap[k//2].key] = k//2
            k=k//2

    def sink(self,k):
        while 2*k<=self.N:
            j=2*k
            if j<self.N and self._heap[j].cost>self._heap[j+1].cost:
                j+=1
            if self._heap[k].cost<=self._heap[j].cost:
                break
            self._heap[k], self._heap[j] = self._heap[j], self._heap[k]
            self._idx[self._heap[k].key] = k
            self._idx[self._heap[j].key] = j
            k=j

    def insert(self,item):
        self._heap.append(item)
        self.N+=1
        self._idx[item.key]=self.N
        self.swim(self.N)

    def pop(self):
        it=self._heap[1]
        if self.N==1:
            self._heap.pop()
            self._idx.pop(it.key)
            self.N -= 1
            return it
        self._heap[1]=self._heap.pop()
        self._idx[self._heap[1].key]=1
        self._idx.pop(it.key)
        self.N-=1
        self.sink(1)
        return it

    def __contains__(self, item):
        if item.key in self._idx:
            return True
        else:
            return False

    def update(self,item):
        it=self._heap[self._idx[item.key]]
        if item.cost<it.cost:
            it.state=item.state
            it.cost = item.cost
            it.ps = item.ps
            it.g=item.g
            it.father = item.father
            self.swim(self._idx[item.key])


class DeliberativePlanner:
    def __init__(self,map):
        self.resolution_pos=1
        self.resolution_time=5
        self.e=4
        self.Ce=5000
        self.Cg_max=500
        self.Wn=1000
        self.Wc=0.5
        self.tmax=125
        self.dmax=100
        self.map=map
        self.control_primitives=load_control_primitives()

    def start(self,s0,sG):
        start_time=time.time()
        print('start')
        s_node=Node(s0,self.state2key(s0),0,0,1)
        self.openlist=OpenList()
        self.closelist=set()
        self.openlist.insert(s_node)
        while self.openlist.N>0:
            sc=self.openlist.pop()
            self.closelist.add(sc.key)
            if (sc.state[0]-sG[0])**2+(sc.state[1]-sG[1])**2<=25:
                return self.generate_total_trajectory(sc)
            current_speed=sc.state[3]
            current_yaw=sc.state[2]
            current_pos = np.array(sc.state[0:2])
            current_time=sc.state[4]
            # print(*sc.state,sc.g,sc.cost)
            # fig.plot(sc.state[1],sc.state[0],"ob",markersize=2)
            # plt.pause(0.0001)
            for ucd in self.control_primitives[current_speed].items():
                pos = current_pos + [ucd[1][-1,0] * cos(current_yaw) - ucd[1][-1,1] * sin(current_yaw), ucd[1][-1,0] * sin(current_yaw) + ucd[1][-1,1] * cos(current_yaw)]
                s1_state=(pos[0],pos[1],yawRange(current_yaw+YAW[ucd[0][1]]),ucd[0][0],current_time+ucd[1][-1,4])
                s1_key=self.state2key(s1_state)
                if s1_key not in self.closelist:
                    s1=Node(s1_state,s1_key)
                    g, Ps=self.cost_to_come(sc,s1,ucd[1][-1,5])
                    if Ps<0.2:
                        continue
                    h=self.cost_to_go(s1,sG)
                    s1.g=g
                    s1.ps=Ps
                    s1.cost=g+self.e*h
                    s1.father=sc
                    # print(g, h,g+self.e*h,s1_state, Ps)
                    if s1 in self.openlist:
                        self.openlist.update(s1)
                        # print('update')
                    else:
                        self.openlist.insert(s1)
            # if time.time()-start_time>20:
            #     break
        return None

    def state2key(self,s_state):
        x,y,yaw,u,t=s_state
        return (int(round(x/self.resolution_pos)*self.resolution_pos),int(round(y/self.resolution_pos)*self.resolution_pos))

    def cost_to_come(self,s,s1,distance):
        Ps=s.ps
        Pcs=self.get_Pcs(s,s1)
        Cs=self.Wn*(self.Wc*(s1.state[4]-s.state[4])/self.tmax+(1.0-self.Wc)*distance/self.dmax)
        g=s.g+Ps/self.Cg_max*((1.0-Pcs)*Cs+Pcs*self.Ce)
        return g,Ps*(1-Pcs)

    def get_Pcs(self,s,s1):
        pos_all=self.compute_trajectory(s.state[0:2],s.state[3:1:-1],s1.state[3:1:-1])
        for pos in pos_all:
            pos_key=(int(round(pos[1]-self.map.offset[0])),int(round(pos[0]-self.map.offset[1])))
            if pos_key[0]<0 or pos_key[0]>=100 or pos_key[1]<0 or pos_key[1]>=100 or self.map.map[pos_key[0],pos_key[1]]==1:
                return 1.0
        return 0.0

    def cost_to_go(self,s1,sG):
        d=np.sqrt((s1.state[0]-sG[0])**2+(s1.state[1]-sG[1])**2)
        t=d/0.8
        return self.Wc*(t/self.tmax)+(1.0-self.Wc)*d/self.dmax

    def generate_total_trajectory(self,s):
        trajectory=[]
        while s is not None:
            trajectory.append(s.state)
            if s.father is None:
                break
            pos_all=self.compute_trajectory(s.father.state[0:2],s.father.state[3:1:-1],s.state[3:1:-1])
            for i in range(len(pos_all)-2,-1,-1):
                trajectory.append(pos_all[i].tolist()+[np.nan]*3)
            s=s.father
        trajectory.reverse()
        return trajectory

    def compute_trajectory(self,s_truepos,s_u_yaw,s1_u_yaw):
        ucd=self.control_primitives[s_u_yaw[0]][(s1_u_yaw[0],"{:.2f}".format(yawRange(s1_u_yaw[1]-s_u_yaw[1])))]
        yaw=s_u_yaw[1]
        pos_all=[]
        pos0 = np.array(s_truepos)
        for i in range(1,ucd.shape[0],2):
            pos_all.append(pos0+[ucd[i,0]*cos(yaw)-ucd[i,1]*sin(yaw),ucd[i,0]*sin(yaw)+ucd[i,1]*cos(yaw)])
        return pos_all


class IntentionModel:
    def __init__(self):
        self.obs_info=[]

    def add_obs(self,info):
        self.obs_info.append(info)

    def forward(self,s_usv,obs):
        obs_new=dict()
        for ob in obs:
            s0=obs[ob]
            yaw,u=s0[2],s0[3]
            obs_new[ob]=s0+[u*cos(yaw),u*sin(yaw),0,0,s_usv[4]+1]
        return obs_new


def load_control_primitives():
    return np.load('control_primitives.npy').item()


def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


if __name__=="__main__":
    # im=IntentionModel()
    # obs={'1':np.array([1,1,0,1,0]),'2':np.array([10,10,1,1,0])}
    # ob_new=im.forward(0,obs)



    dp=DeliberativePlanner(map)
    s0=tuple(np.array((5,5,0,0.8,0),dtype=np.float64))
    sG=tuple(np.array((80,80,0,0.4,0),dtype=np.float64))
    start_time=time.time()
    tra=np.array(dp.start(s0,sG))
    print("runtime is {}".format(time.time()-start_time))

    fig.plot(tra[:,1],tra[:,0])
    for i in range(tra.shape[0]):
        if not np.isnan(tra[i,4]):
            fig.plot(tra[i,1],tra[i,0],"ob",markersize=2)
    fig.plot(sG[1],sG[0],"ob",markersize=5)
    plt.show()



