from numpy import sin,cos,pi,exp
from map import Map,DynamicObstacle,generate_do_trajectory
import numpy as np
import time
import os
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt


YAW={'0.79':pi/4,'0.52':pi/6,'0.26':pi/12,'-0.79':-pi/4,'-0.52':-pi/6,'-0.26':-pi/12,'0.00':0,'1.05':pi/3,'-1.05':-pi/3,'0.17':pi/18,'-0.17':-pi/18,'1.57':pi/2,'-1.57':-pi/2}




class Node():
    def __init__(self,state,key,g=None,cost=None,ps=None,father=None):
        #state=(N,E,yaw,u,t)
        self.state=state
        self.key=key
        self.cost=cost
        self.ps=ps
        self.g=g
        self.father=father
        self.encounter_type=dict()



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
    def __init__(self,map,resolution_pos,tmax,dmax,resolution_time,default_speed,e=1.5):
        self.resolution_pos=resolution_pos
        self.resolution_time=resolution_time
        self.e=e
        self.Ce=500
        self.Cg_max=1000
        self.Wn=1000
        self.Wc=0.5
        self.default_speed=default_speed
        self.tmax=tmax
        self.dmax=dmax
        self.gamma=0.01
        self.C_sigma=0.25
        self.map=map
        self.control_primitives=load_control_primitives()
        self.do_tra=None
        self.local_radius=80
        self.tcpa_min=50
        self.dcpa_min=20
        self.collision_risk_ob=dict()
        self.local_range_ob=[]

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
            current_yaw=sc.state[2]
            current_pos = np.array(sc.state[0:2])
            current_time=sc.state[4]
            current_speed=sc.state[3]
            if self.do_tra:
                self.evaluate_encounter(sc)
            print(self.collision_risk_ob)

            evaluate_node+=1


            for ucd in self.control_primitives[current_speed].items():
                pos = current_pos + [ucd[1][-1,0] * cos(current_yaw) - ucd[1][-1,1] * sin(current_yaw), ucd[1][-1,0] * sin(current_yaw) + ucd[1][-1,1] * cos(current_yaw)]
                if self.collision_with_static_ob(pos)==1.0:
                    continue
                s1_state=(pos[0],pos[1],yawRange(current_yaw+ucd[0][1]*pi/180),self.default_speed,current_time+ucd[1][-1,4])
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
                        # if fig:
                        #     fig.plot(s1_state[1],s1_state[0],'ob',markersize=1)

            if fig:

                try:
                    for plot_line in plot_lines:
                        fig.lines.remove(plot_line[0])
                except:
                    pass

                try:
                    for plot_patch in plot_patches:
                        fig.patches.remove(plot_patch)
                except:
                    pass

                plot_lines = []
                plot_patches=[]
                # plot_lines.append(fig.plot(sc.state[1], sc.state[0], "or", markersize=5))
                fig.plot(sc.key[1], sc.key[0], "ob", markersize=1)
                fig.add_patch(patches.Arrow(sc.key[1], sc.key[0],0.5*sin(current_yaw),0.5*cos(current_yaw), width=0.5))
                if self.do_tra is not None:
                    for key in self.do_tra:
                        if int(sc.state[4])+10<self.do_tra[key].shape[0]:
                            do_y,do_x,do_yaw=self.do_tra[key][int(sc.state[4]),1],self.do_tra[key][int(sc.state[4]),0],self.do_tra[key][int(sc.state[4]),2]
                            do_y_next, do_x_next = self.do_tra[key][int(sc.state[4])+10, 1], self.do_tra[key][int(sc.state[4])+10, 0]
                            plot_lines.append(fig.plot(do_y,do_x,"or",markersize=5))
                            plot_lines.append(fig.plot([do_y,do_y_next],[do_x,do_x_next],'--r'))
                            if key in self.collision_risk_ob:
                                plot_patches.append(plot_colrges_cost_range(do_x,do_y,do_yaw,self.collision_risk_ob[key][0],fig))
                            plot_lines.append(plot_circle((do_y,do_x),np.sqrt(sc.state[4]*self.C_sigma),fig))
                # if sc.father is not None:
                #     fig.plot([sc.state[1],sc.father.state[1]],[sc.state[0],sc.father.state[0]],"--b")
                plt.pause(0.0001)
        return None

    def state2key(self,s_state):
        x,y,yaw,u,t=s_state
        return (int(round(x/self.resolution_pos)*self.resolution_pos),int(round(y/self.resolution_pos)*self.resolution_pos))

    def cost_to_come(self,s,s1,distance):
        Pcs,colrges_break=self.evaluate_primitive(s,s1)
        # print(np.int((s1.state[2]-s.state[2])*180/pi),Pcs)
        # colrges_break=0
        Ps=s.ps
        Cs=self.Wn*(self.Wc*(s1.state[4]-s.state[4])/self.tmax+(1.0-self.Wc)*distance/self.dmax+colrges_break)
        g=s.g+Ps/self.Cg_max*((1.0-Pcs)*Cs+Pcs*self.Ce)
        return g,Ps*(1-Pcs)

    # def get_pcs(self,pos_primitive,t):
    #     Pcsu=0.0
    #     for pos in pos_primitive:
    #         t+=self.resolution_time
    #         p=self.collision_with_static_ob(pos)
    #         if p==1.0:
    #             return 1.0
    #         Pcsu=max(self.collision_with_dynamic_ob(pos,t),Pcsu)
    #     # Pcs=exp(-self.gamma*s1.state[4])*Pcsu
    #     return Pcsu

    def evaluate_primitive(self,s,s1):
        primitives = self.compute_trajectory(s.state, s1.state)
        t=int(s.state[4])
        colrges_break=0.0
        no_Pcsu=1.0
        for i,primitive in enumerate(primitives):
            t+=self.resolution_time
            p=self.collision_with_static_ob(primitive[0:2])
            if p==1.0:
                return 1.0,0.0
            no_Pcsu_t=1.0
            if (i+1)%(primitives.shape[0]//2)==0:
                for key in self.collision_risk_ob:
                    colrges_break += self.colrges_cost(primitive, self.do_tra[key][t], key)
                for key in self.local_range_ob:
                    if key not in self.collision_risk_ob:
                        pos_do = self.do_tra[key][t, 0:2]
                    elif self.collision_risk_ob[key][0]=="head on":
                        pos_do=self.do_tra[key][t,0:2]+5*np.array([-sin(self.do_tra[key][t,2]),cos(self.do_tra[key][t,2])])
                    else:
                        pos_do = self.do_tra[key][t, 0:2]
                    distance=np.sqrt(np.inner(pos_do-primitive[0:2],pos_do-primitive[0:2]))
                    no_Pcsu_t *= (1 - collision_pro_cal(distance,self.C_sigma*t,6))
                no_Pcsu *=no_Pcsu_t
        return (1-no_Pcsu)*exp(-self.gamma*s.state[4]),colrges_break/2

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
            pos_all=self.compute_trajectory(s.father.state,s.state)
            for i in range(len(pos_all)-2,-1,-1):
                trajectory.append(pos_all[i])
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
        trajectory=[]
        while j<len(states):

            if abs(states[j][2]-states[i][2])<0.3 and abs(states[j][2]-states[j-1][2])<0.3 and states[j][2]-states[j-1][2]!=0.0:
                j+=1
            else:
                if j-i<=2:
                    for ik in range(i,j):
                        trajectory.append(states[ik])
                        pos_all=self.compute_trajectory(states[ik],states[ik+1])
                        for k in range(len(pos_all)-1):
                            trajectory.append(pos_all[k])
                    i=j
                    j+=1
                else:
                    trajectory.append(states[i])
                    dY,dX=states[j-1][1]-states[i][1],states[j-1][0]-states[i][0]
                    head=np.arctan2(dY,dX)
                    for ik in range(self.resolution_time,np.int(states[j-1][4])-np.int(states[i][4]),self.resolution_time):
                        trajectory.append([states[i][0]+dX*ik/(states[j-1][4]-states[i][4]),states[i][1]+dY*ik/(states[j-1][4]-states[i][4]),head,states[i][3],states[i][4]+ik])
                    # trajectory.append(states[j-1])
                    i=j-1
        trajectory.append(states[i])
        if i!=j-1:
            dY, dX = states[j - 1][1] - states[i][1], states[j - 1][0] - states[i][0]
            head = np.arctan2(dY, dX)
            for ik in range(self.resolution_time, np.int(states[j - 1][4]) - np.int(states[i][4]),
                            self.resolution_time):
                trajectory.append([states[i][0] + dX * ik / (states[j - 1][4] - states[i][4]),
                                   states[i][1] + dY * ik / (states[j - 1][4] - states[i][4]), head, states[i][3],
                                   states[i][4] + ik])
            # trajectory.append(states[j - 1])

        return trajectory



    # def compute_trajectory(self,s_truepos,s_u_yaw,s1_u_yaw):
    #     ucd=self.control_primitives[s_u_yaw[0]][(s1_u_yaw[0],"{:.2f}".format(yawRange(s1_u_yaw[1]-s_u_yaw[1])))]
    #     yaw=s_u_yaw[1]
    #     pos_all=[]
    #     pos0 = np.array(s_truepos)
    #     for i in range(self.resolution_time-1,ucd.shape[0],self.resolution_time):
    #         pos_all.append(pos0+[ucd[i,0]*cos(yaw)-ucd[i,1]*sin(yaw),ucd[i,0]*sin(yaw)+ucd[i,1]*cos(yaw)])
    #     return pos_all

    def compute_trajectory(self,s_state,s1_state):
        ucd = self.control_primitives[s_state[3]][
            (int(s1_state[4] - s_state[4]), np.int(np.round(180/pi*yawRange(s1_state[2] - s_state[2]))))]
        yaw=s_state[2]
        state_all=[]
        for i in range(self.resolution_time-1,ucd.shape[0],self.resolution_time):
            state_all.append([s_state[0]+ucd[i,0]*cos(yaw)-ucd[i,1]*sin(yaw),s_state[1]+ucd[i,0]*sin(yaw)+ucd[i,1]*cos(yaw),yawRange(yaw+ucd[i,2]),ucd[i,3],s_state[4]+ucd[i,4]])
        return np.array(state_all)

    def collision_with_static_ob(self,pos):
        pos_key = (int(round((pos[1] - self.map.offset[0])/self.map.resolution)), int(round((pos[0] - self.map.offset[1])/self.map.resolution)))
        if pos_key[0] < 0 or pos_key[0] >= self.map.size[1] or pos_key[1] < 0 or pos_key[1] >= self.map.size[0] or self.map.map[
            pos_key[0], pos_key[1]] == 1:
            return 1.0
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

    def evaluate_encounter(self,sc):
        if sc.father is not None:
            self.collision_risk_ob = copy.deepcopy(sc.father.encounter_type)
        for key in list(self.collision_risk_ob.keys()):
            if self.collision_risk_ob[key][1]==5:
                self.collision_risk_ob.pop(key)
            else:
                self.collision_risk_ob[key][1]+=1

        pos=np.array(sc.state[:2])
        yaw,u,t=sc.state[2:]
        t=int(t)
        self.local_range_ob=[key for key,value in self.do_tra.items() if t<value.shape[0]-10 and np.inner(value[t][0:2]-pos,value[t][0:2]-pos)<self.local_radius**2]
        # self.collision_risk_ob=dict()
        for key in self.local_range_ob:
            tcpa,dcpa=get_cpa(self.do_tra[key][t],sc.state)
            # print(tcpa,dcpa)
            if tcpa>0 and tcpa<self.tcpa_min and dcpa<self.dcpa_min:
                encounter_type=colrges_encounter_type(sc.state,self.do_tra[key][t])
                if encounter_type=='head on':
                    self.collision_risk_ob[key]=['head on',0]
                elif encounter_type=='cross from right':
                    if key not in self.collision_risk_ob or self.collision_risk_ob[key][0]=='cross from right':
                        self.collision_risk_ob[key] = ['cross from right', 0]
        sc.encounter_type=self.collision_risk_ob


    def colrges_cost(self,s_usv,s_ob,key):
        x1,y1,yaw1,u1,_=s_usv
        x2,y2,yaw2,u2,_=s_ob
        alpha_b=yawRange(np.arctan2(y1-y2,x1-x2)-yaw2)
        distance=np.sqrt(np.inner(s_usv[0:2]-s_ob[0:2],s_usv[0:2]-s_ob[0:2]))
        if self.collision_risk_ob[key][0]=="head on" and alpha_b>-pi/24 and alpha_b<pi/2 and distance<30:
            return 1.0
        elif self.collision_risk_ob[key][0]=="cross from right" and alpha_b>-pi/4 and alpha_b<pi/8 and distance<30:
            return 1.0
        else:
            return 0.0




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
        step = (2*d) / 10
        step1=(r-d)/10
        x=np.arange(d+r-0.1,r-d,-step)
        x1=np.arange(0,r-d,step1)
        s=np.sum(np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d)*2*x*step / (2 * pi * sigma2) * exp(-1 / 2 * x ** 2 / sigma2))
        s1=np.sum(1 / (2 * pi * sigma2) * exp(-1 / 2 * x1 ** 2 / sigma2)*2*pi*x1*step1)
        return s+s1

def load_control_primitives():
    print("load control primitives from {}".format(os.curdir))
    return np.load('control_primitives.npy').item()


def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x

def plot_circle(c,r,fig):
    theta = np.linspace(0, 2 * np.pi, 800)
    y, x = np.cos(theta) * r+c[0], np.sin(theta) * r+c[1]
    circle=fig.plot(y, x, "--r")
    return circle

def plot_colrges_cost_range(x,y,yaw,encounter_type,fig):
    if encounter_type=='head on':
        theta=-yaw*180/pi
        wedge=patches.Wedge((y,x),30,theta,theta+97.5,color='y')
        return fig.add_patch(wedge)
    elif encounter_type=='cross from right':
        theta=-yaw*180/pi+90
        wedge=patches.Wedge((y,x),30,theta-22.5,theta+45,color='y')
        return fig.add_patch(wedge)

def get_cpa(s1,s2):
    x1,y1,yaw1,u1,_=s1
    x2,y2,yaw2,u2,_=s2
    dv=np.array([u1*cos(yaw1)-u2*cos(yaw2),u1*sin(yaw1)-u2*sin(yaw2)])
    dpos=np.array([x1-x2,y1-y2])
    tcpa=-np.inner(dv,dpos)/np.inner(dv,dv)
    dpos1=dpos+tcpa*dv
    dcpa=np.sqrt(np.inner(dpos1,dpos1))
    return tcpa,dcpa

def colrges_encounter_type(s_usv,s_ob):
    x1,y1,yaw1,u1,_=s_usv
    x2,y2,yaw2,u2,_=s_ob
    alpha_b=yawRange(np.arctan2(y1-y2,x1-x2)-yaw2)
    alpha_h=yawRange(yaw1-yaw2)
    if abs(alpha_b)<=pi/12 and abs(alpha_h)>=11*pi/12:
        encounter_type="head on"
    elif alpha_b>pi/12 and alpha_b<3*pi/4 and alpha_h>-11*pi/12 and alpha_h<-pi/4:
        encounter_type="cross from left"
    elif alpha_b>-3*pi/4 and alpha_b<-pi/12 and alpha_h>pi/4 and alpha_h<11*pi/12:
        encounter_type="cross from right"
    # elif abs(alpha_b)>=3*pi/4 and abs(alpha_h)<=pi/4:
    #     encounter_type="take over"
    else:
        encounter_type=None
    return encounter_type



def test_head_on():
    #地图、起点、目标
    map_size = (100, 100)
    map = Map()
    map.new_map(map_size,resolution=1)
    fig = plt.gca()
    fig.axis([0, map_size[0], 0, map_size[1]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0=tuple(np.array((15,15,pi/4,0.8,0),dtype=np.float64))
    sG=tuple(np.array((100,100,pi,0.8,0),dtype=np.float64))
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    #静态障碍物
    # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    # rectangle_static_obstacles=((40,75,70,50),(150,75,100,50),(175,175,60,60))
    rectangle_static_obstacles = ()


    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle", config=ob)
        rect = patches.Rectangle((ob[0], ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)

    dp = DeliberativePlanner(map, 1, 125, 100, 1, 0.8,1.2)
    #动态障碍物
    # do_tra=np.array(dp.start(sG, s0))
    do_tra=generate_do_trajectory(95, 100, -3 * pi / 4, 0.70, 200)
    do_tra_dic={"do1":do_tra}
    fig.plot(do_tra[:, 1], do_tra[:, 0], "y")


    start_time=time.time()
    dp.set_dynamic_obstacle(do_tra_dic)
    tra=np.array(dp.start(s0,sG,fig))
    print("runtime is {},closelist node number is {},trajectory total time is {}".format(time.time() - start_time,
                                                                                         len(dp.closelist),
                                                                                         tra[-1, -1]))

    fig.plot(tra[:,1],tra[:,0],"r")
    for i in range(tra.shape[0]):
        if np.int(np.round(tra[i,2]*180/pi))%15==0  and tra[i, 3] == 0.8:
            fig.plot(tra[i,1],tra[i,0],"or",markersize=2)

    # a = np.array(list(dp.closelist), dtype=np.float64)
    # fig.plot(a[:,1],a[:,0],'ob',markersize=1)

    plt.show()
    return tra

def test_cross():
    #地图、起点、目标
    map_size = (100, 100)
    map = Map()
    map.new_map(map_size,resolution=1)
    fig = plt.gca()
    fig.axis([0, map_size[0], 0, map_size[1]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0=tuple(np.array((10,5,pi/4,0.8,0),dtype=np.float64))
    sG=tuple(np.array((80,80,pi,0.8,0),dtype=np.float64))
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    #静态障碍物
    # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    # rectangle_static_obstacles=((40,75,70,50),(150,75,100,50),(175,175,60,60))
    rectangle_static_obstacles = ()


    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle", config=ob)
        rect = patches.Rectangle((ob[0], ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)

    #动态障碍物

    do_tra=generate_do_trajectory(10,95,-0.7,0.75,200)
    do_tra_dic = {"do1": do_tra}
    fig.plot(do_tra[:, 1], do_tra[:, 0], "y")

    dp=DeliberativePlanner(map,1,125,100,1,0.8,1.2)
    start_time=time.time()
    dp.set_dynamic_obstacle(do_tra_dic)
    tra=np.array(dp.start(s0,sG,fig))
    print("runtime is {},closelist node number is {},trajectory total time is {}".format(time.time() - start_time,
                                                                                         len(dp.closelist),
                                                                                         tra[-1, -1]))

    fig.plot(tra[:,1],tra[:,0],"r")
    for i in range(tra.shape[0]):
        if np.int(np.round(tra[i,2]*180/pi))%15==0  and tra[i, 3] == 0.8:
            fig.plot(tra[i,1],tra[i,0],"or",markersize=2)
            fig.plot(do_tra[i,1],do_tra[i,0],'ob',markersize=2)

    # a = np.array(list(dp.closelist), dtype=np.float64)
    # fig.plot(a[:,1],a[:,0],'ob',markersize=1)

    plt.show()
    return tra

def test_static():
    # 地图、起点、目标
    map_size = (300, 300)
    map = Map()
    map.new_map(map_size, resolution=1)
    fig = plt.gca()
    fig.axis([0, map_size[0], 0, map_size[1]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0 = tuple(np.array((25, 25, pi/4, 0.8, 0), dtype=np.float64))
    sG = tuple(np.array((280, 280, -pi/2, 0.8, 0), dtype=np.float64))
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    # 静态障碍物
    # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # rectangle_static_obstacles = ((0, 50, 60, 10), (50, 0, 10, 48))
    # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    rectangle_static_obstacles=((40,42,70,75),(150,75,100,50),(175,175,60,60))
    # rectangle_static_obstacles = ()

    for ob in rectangle_static_obstacles:
        map.add_static_obstacle(type="rectangle", config=ob)
        rect = patches.Rectangle((ob[0], ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)

    dp = DeliberativePlanner(map, 1, 125, 100, 1, 0.8,1.5)
    # 动态障碍物

    s0_do = tuple(np.array((40, 200, -pi/2, 0.8, 0), dtype=np.float64))
    sG_do = tuple(np.array((40, 20, pi/2, 0.8, 0), dtype=np.float64))
    # do_tra=np.array(dp.start(s0_do, sG_do))
    # do_tra_dic = {"do1": do_tra}
    # fig.plot(do_tra[:, 1], do_tra[:, 0], "y")

    start_time = time.time()
    # dp.set_dynamic_obstacle(do_tra_dic)
    tra = np.array(dp.start(s0, sG))
    print("runtime is {},closelist node number is {},trajectory total time is {}".format(time.time() - start_time,len(dp.closelist),tra[-1, -1]))
    fig.plot(tra[:, 1], tra[:, 0], "r")
    for i in range(tra.shape[0]):
        if np.int(np.round(tra[i,2]*180/pi))%15==0  and tra[i, 3] == 0.8:
            fig.plot(tra[i, 1], tra[i, 0], "or", markersize=2)

    # a = np.array(list(dp.closelist), dtype=np.float64)
    # fig.plot(a[:,1],a[:,0],'ob',markersize=1)

    plt.show()
    return tra


if __name__=="__main__":
    # #地图、起点、目标
    # map_size = (300, 300)
    # map = Map()
    # map.new_map(map_size,resolution=1)
    # fig = plt.gca()
    # fig.axis([0, map_size[0], 0, map_size[1]])
    # fig.set_xlabel('E/m')
    # fig.set_ylabel('N/m')
    # s0=tuple(np.array((50,50,0,0.8,0),dtype=np.float64))
    # sG=tuple(np.array((250,250,pi,0.8,0),dtype=np.float64))
    # fig.plot(sG[1], sG[0], "ob", markersize=5)
    #
    # #静态障碍物
    # # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    # rectangle_static_obstacles=((40,75,70,50),(150,75,100,50),(175,175,60,60))
    #
    #
    # for ob in rectangle_static_obstacles:
    #     map.add_static_obstacle(type="rectangle", config=ob)
    #     rect = patches.Rectangle((ob[0]+3, ob[1]+3), ob[2]-6, ob[3]-6, color='y')
    #     fig.add_patch(rect)
    #
    # #动态障碍物
    # # do_tra={"do1":generate_do_trajectory(50,100,-pi/2,0.70,200)}
    #
    #
    # dp=DeliberativePlanner(map,1,500,400,1,0.8)
    # start_time=time.time()
    # dp.set_dynamic_obstacle(dict())
    # tra=np.array(dp.start(s0,sG))
    # print("runtime is {}".format(time.time()-start_time))
    #
    # fig.plot(tra[:,1],tra[:,0],"r")
    # for i in range(tra.shape[0]):
    #     if i%10==0:
    #         fig.plot(tra[i,1],tra[i,0],"or",markersize=2)
    #
    # # a = np.array(list(dp.closelist), dtype=np.float64)
    # # fig.plot(a[:,1],a[:,0],'ob',markersize=1)
    #
    # plt.show()
    tra=test_cross()
    # tra=test_head_on()
    # tra=test_static()



