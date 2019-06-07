from numpy import sin,cos,pi,exp
from map import Map,DynamicObstacle,generate_do_trajectory
import numpy as np
import time
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt


YAW={'0.79':pi/4,'0.52':pi/6,'0.26':pi/12,'-0.79':-pi/4,'-0.52':-pi/6,'-0.26':-pi/12,'0.00':0,'1.05':pi/3,'-1.05':-pi/3,'0.17':pi/18,'-0.17':-pi/18}



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
        it.state=item.state
        it.cost = item.cost
        it.ps = item.ps
        it.g=item.g
        it.father = item.father
        self.swim(self._idx[item.key])


class DeliberativePlanner:
    def __init__(self,map,resolution_pos,tmax,dmax,resolution_time,default_speed):
        self.resolution_pos=resolution_pos
        self.resolution_time=resolution_time
        self.e=1.2
        self.Ce=1000
        self.Cg_max=1000
        self.Wn=1000
        self.Wc=0.5
        self.default_speed=default_speed
        self.tmax=tmax
        self.dmax=dmax
        self.gamma=0.1
        self.C_std=0.1
        self.map=map
        self.control_primitives=load_control_primitives()
        self.do_tra=None

    def set_dynamic_obstacle(self,do_tra,do_config=None):
        self.do_config=do_config
        self.do_tra=do_tra

    def start(self,s0,sG,fig=None):
        print('start')
        evaluate_node=0
        s_node=Node(s0,self.state2key(s0),0,0,1)
        self.openlist=OpenList()
        self.closelist=set()
        self.openlist.insert(s_node)
        while self.openlist.N>0:
            sc=self.openlist.pop()
            self.closelist.add(sc.key)
            if (sc.state[0]-sG[0])**2+(sc.state[1]-sG[1])**2<=(self.resolution_pos*5)**2:
                return self.generate_total_trajectory1(sc)
            current_speed=sc.state[3]
            current_yaw=sc.state[2]
            current_pos = np.array(sc.state[0:2])
            current_time=sc.state[4]

            evaluate_node+=1
            print(evaluate_node)

            if fig:
                try:
                    for plot_item in plot_items:
                        fig.lines.remove(plot_item[0])
                except:
                    pass
                fig.plot(sc.state[1],sc.state[0],"ob",markersize=1)
                plot_items = []
                if self.do_tra is not None:
                    for key in self.do_tra:
                        do_y,do_x=self.do_tra[key][int(sc.state[4]),1],self.do_tra[key][int(sc.state[4]),0]
                        plot_items.append(fig.plot(do_y,do_x,"or",markersize=5))
                        plot_items.append(plot_circle((do_y,do_x),sc.state[4]*self.C_std))
                if sc.father is not None:
                    fig.plot([sc.state[1],sc.father.state[1]],[sc.state[0],sc.father.state[0]],"b")
                plt.pause(0.0001)

            for ucd in self.control_primitives[current_speed].items():
                pos = current_pos + [ucd[1][-1,0] * cos(current_yaw) - ucd[1][-1,1] * sin(current_yaw), ucd[1][-1,0] * sin(current_yaw) + ucd[1][-1,1] * cos(current_yaw)]
                if self.collision_with_static_ob(pos)==1.0:
                    continue
                s1_state=(pos[0],pos[1],yawRange(current_yaw+YAW[ucd[0][1]]),ucd[0][0],current_time+ucd[1][-1,4])
                s1_key=self.state2key(s1_state)
                if not s1_key in self.closelist:
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
                        if s1.cost < self.openlist._heap[self.openlist._idx[s1.key]].cost:
                            # test_dic.setdefault(s1_key, [self.openlist._heap[self.openlist._idx[s1_key]].state[2:]]).append(s1_state[2:])
                            self.openlist.update(s1)
                    else:
                        self.openlist.insert(s1)
        return None

    def state2key(self,s_state):
        x,y,yaw,u,t=s_state
        return (int(round(x/self.resolution_pos)*self.resolution_pos),int(round(y/self.resolution_pos)*self.resolution_pos))

    def cost_to_come(self,s,s1,distance):
        Ps=s.ps
        Pcs=self.get_pcs(s,s1)
        # print(Pcs)
        Cs=self.Wn*(self.Wc*(s1.state[4]-s.state[4])/self.tmax+(1.0-self.Wc)*distance/self.dmax)
        g=s.g+Ps/self.Cg_max*((1.0-Pcs)*Cs+Pcs*self.Ce)
        return g,Ps*(1-Pcs)

    def get_pcs(self,s,s1):
        pos_all=self.compute_trajectory(s.state[0:2],s.state[3:1:-1],s1.state[3:1:-1])
        Pcsu=0.0
        Pcob=0.0
        t=s.state[4]
        for pos in pos_all:
            t+=self.resolution_time
            p=self.collision_with_static_ob(pos)
            if p==1.0:
                return 1.0
            Pcob=max(p,Pcob)
            Pcsu=max(self.collision_with_dynamic_ob(pos,t),Pcsu)
        # Pcs=exp(-self.gamma*s1.state[4])*Pcsu
        return 1-(1-Pcsu)*(1-Pcob)

    def cost_to_go(self,s1,sG):
        d=np.sqrt((s1.state[0]-sG[0])**2+(s1.state[1]-sG[1])**2)
        t=d/self.default_speed
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

    def generate_total_trajectory1(self,s):
        states=[]
        while s is not None:
            states.append(s.state)
            s=s.father
        states.reverse()
        i,j=0,1
        trajectory=[states[0]]
        while j<len(states):

            if abs(states[j][2]-states[i][2])<0.3 and abs(states[j][2]-states[j-1][2])<0.3 and states[j][2]-states[j-1][2]!=0.0:
                j+=1
            else:
                if j-i<=2:
                    for ik in range(i,j):
                        trajectory.append(states[ik])
                        pos_all=self.compute_trajectory(states[ik][0:2],states[ik][3:1:-1],states[ik+1][3:1:-1])
                        for k in range(len(pos_all)-1):
                            trajectory.append(pos_all[k].tolist() + [np.nan] * 3)
                    i=j
                    j+=1
                else:
                    trajectory.append(states[i])
                    trajectory.append(states[j-1])
                    i=j-1
        trajectory.append(states[i])
        if i!=j-1:
            trajectory.append(states[j-1])

        return trajectory



    def compute_trajectory(self,s_truepos,s_u_yaw,s1_u_yaw):
        ucd=self.control_primitives[s_u_yaw[0]][(s1_u_yaw[0],"{:.2f}".format(yawRange(s1_u_yaw[1]-s_u_yaw[1])))]
        yaw=s_u_yaw[1]
        pos_all=[]
        pos0 = np.array(s_truepos)
        for i in range(self.resolution_time-1,ucd.shape[0],self.resolution_time):
            pos_all.append(pos0+[ucd[i,0]*cos(yaw)-ucd[i,1]*sin(yaw),ucd[i,0]*sin(yaw)+ucd[i,1]*cos(yaw)])
        return pos_all

    def collision_with_static_ob(self,pos):
        pos_key = (int(round((pos[1] - self.map.offset[0])/self.map.resolution)), int(round((pos[0] - self.map.offset[1])/self.map.resolution)))
        if pos_key[0] < 0 or pos_key[0] >= self.map.size[1] or pos_key[1] < 0 or pos_key[1] >= self.map.size[0] or self.map.map[
            pos_key[0], pos_key[1]] == 1:
            return 1.0
        # a=self.map.map[pos_key[0]-3:pos_key[0]+3,pos_key[1]-3:pos_key[1]+3]
        # return np.count_nonzero(a)/a.size/100
        return 0.0


    def collision_with_dynamic_ob(self,pos,t):
        no_pcsu=1.0
        t=int(t)
        if self.do_tra is not None:
            for key in self.do_tra:
                if t>=self.do_tra[key].shape[0]:
                    continue
                d_pos=pos-self.do_tra[key][int(t),0:2]
                stdx,stdy=t*self.C_std,t*self.C_std
                if np.inner(d_pos,d_pos)<3**2:
                    return 1.0
                p=1/(2*pi*stdx*stdy)*exp(-1/2*((d_pos[0]/stdx)**2+(d_pos[1]/stdy)**2))*40
                no_pcsu*=1-p
        return 1-no_pcsu



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
    print("load control primitives from {}".format(os.curdir))
    return np.load('control_primitives.npy').item()


def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def plot_circle(c,r):
    theta = np.linspace(0, 2 * np.pi, 800)
    y, x = np.cos(theta) * r+c[0], np.sin(theta) * r+c[1]
    circle=fig.plot(y, x, "--r")
    return circle




if __name__=="__main__":
    #地图、起点、目标
    map_size = (300, 300)
    map = Map()
    map.new_map(map_size,resolution=1)
    fig = plt.gca()
    fig.axis([0, map_size[0], 0, map_size[1]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0=tuple(np.array((50,50,0,0.8,0),dtype=np.float64))
    sG=tuple(np.array((250,250,pi,0.8,0),dtype=np.float64))
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    #静态障碍物
    # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    rectangle_static_obstacles=((40,75,70,50),(150,75,100,50),(175,175,60,60))


    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle", config=ob)
        rect = patches.Rectangle((ob[0]+3, ob[1]+3), ob[2]-6, ob[3]-6, color='y')
        fig.add_patch(rect)

    #动态障碍物
    # do_tra={"do1":generate_do_trajectory(50,100,-pi/2,0.70,200)}


    dp=DeliberativePlanner(map,1,500,400,1,0.8)
    start_time=time.time()
    # dp.set_dynamic_obstacle(do_tra)
    tra=np.array(dp.start(s0,sG))
    print("runtime is {}".format(time.time()-start_time))

    fig.plot(tra[:,1],tra[:,0],"r")
    for i in range(tra.shape[0]):
        if not np.isnan(tra[i,4]):
            fig.plot(tra[i,1],tra[i,0],"or",markersize=2)

    # a = np.array(list(dp.closelist), dtype=np.float64)
    # fig.plot(a[:,1],a[:,0],'ob',markersize=1)

    plt.show()



