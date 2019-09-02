from numba import jit
from numpy import sin, cos, pi, exp
import numpy as np
import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt

FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

Wn = 1000
Wc = 0.5
Cg_max = 1000
Ce = 500
gamma = 0.01
# gamma=0
C_sigma = 0.25
C_colrges=5
dimension=2
# {'None':1,'cross from left':0,'take over':2,'cross from right':3,'head on':4}


class Node:
    def __init__(self, state, key, g=None, cost=None, ps=None, father=None):
        # state=(N,E,yaw,u,t)
        self.state = state
        self.key = key
        self.cost = cost
        self.ps = ps
        self.g = g
        self.father = father
        self.encounter_type = np.zeros((0, 3), dtype=np.int)


class OpenList:
    def __init__(self):
        self._heap = [None]
        self.N = 0
        self._idx = dict()

    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self._heap[1:])

    def swim(self, k):
        while k > 1 and self._heap[k // 2].cost > self._heap[k].cost:
            self._heap[k], self._heap[k // 2] = self._heap[k // 2], self._heap[k]
            self._idx[self._heap[k].key] = k
            self._idx[self._heap[k // 2].key] = k // 2
            k = k // 2

    def sink(self, k):
        while 2 * k <= self.N:
            j = 2 * k
            if j < self.N and self._heap[j].cost > self._heap[j + 1].cost:
                j += 1
            if self._heap[k].cost <= self._heap[j].cost:
                break
            self._heap[k], self._heap[j] = self._heap[j], self._heap[k]
            self._idx[self._heap[k].key] = k
            self._idx[self._heap[j].key] = j
            k = j

    def insert(self, item):
        self._heap.append(item)
        self.N += 1
        self._idx[item.key] = self.N
        self.swim(self.N)

    def pop(self):
        it = self._heap[1]
        if self.N == 1:
            self._heap.pop()
            self._idx.pop(it.key)
            self.N -= 1
            return it
        self._heap[1] = self._heap.pop()
        self._idx[self._heap[1].key] = 1
        self._idx.pop(it.key)
        self.N -= 1
        self.sink(1)
        return it

    def __contains__(self, item):
        if item.key in self._idx:
            return True
        else:
            return False

    def update(self, item):
        it = self._heap[self._idx[item.key]]
        it.state = item.state
        it.cost = item.cost
        it.ps = item.ps
        it.g = item.g
        it.father = item.father
        self.swim(self._idx[item.key])


class DeliberativePlanner:
    def __init__(
            self,
            static_map,
            resolution_pos,
            resolution_time,
            default_speed,
            primitive_file_path,
            e=1.5):
        self.map = static_map
        self.resolution_pos = resolution_pos
        self.resolution_time = resolution_time
        self.e = e
        self.default_speed = default_speed
        self.dmax = np.sum(static_map.size) * static_map.resolution / 2
        self.tmax = self.dmax / self.default_speed
        self.control_primitives = load_control_primitives(primitive_file_path)
        self.do_tra = np.zeros((0, 0, 5))
        self.local_radius = 100
        self.tcpa_min = 50
        self.dcpa_min = 50
        self.collision_risk_ob = np.zeros((0, 3), dtype=np.int)
        # self.local_range_ob=[]
        logging.info('resolution_pos:{},resolution_time:{},e:{},default_speed:{},local_radius:{},tcpa_min:{},dcpa_min:{}'
                     .format(resolution_pos,resolution_time,e,default_speed,self.local_radius,self.tcpa_min,self.dcpa_min))

    def set_dynamic_obstacle(self, do_tra, do_config=None):
        self.do_config = do_config
        self.do_tra = do_tra

    def start(self, s0, sG, fig=None):
        # logging.info('start')
        evaluate_node = 0
        s_node = Node(s0, self.state2key(s0), 0, 0, 1)
        self.openlist = OpenList()
        self.closelist = set()
        self.openlist.insert(s_node)
        while self.openlist.N > 0:
            sc = self.openlist.pop()
            self.closelist.add(sc.key)
            if (sc.state[0] - sG[0])**2 + (sc.state[1] - sG[1])**2 <= (self.resolution_pos * 5)**2:
                return self.generate_total_trajectory1(sc)
            current_yaw = sc.state[2]
            current_pos = np.array(sc.state[0:2])
            current_time = sc.state[4]
            current_speed = sc.state[3]
            self.evaluate_encounter(sc)
            # print(self.collision_risk_ob)
            evaluate_node += 1
            for ucd in self.control_primitives[current_speed].items():
                pos = current_pos + [ucd[1][-1, 0] * cos(current_yaw) - ucd[1][-1, 1] * sin(current_yaw),
                                     ucd[1][-1, 0] * sin(current_yaw) + ucd[1][-1, 1] * cos(current_yaw)]
                if collision_with_static_ob(
                        pos,
                        self.map.map,
                        self.map.offset,
                        self.map.resolution) == 1.0:
                    continue
                s1_state = (pos[0],
                            pos[1],
                            yawRange(current_yaw + ucd[0][1] * pi / 180),
                            self.default_speed,
                            current_time + ucd[1][-1,4])
                s1_key = self.state2key(s1_state)
                if s1_key not in self.closelist:
                    g, Ps = cost_to_come(sc.state, s1_state, ucd[1], sc.ps, sc.g, self.map.map, self.map.offset, self.map.resolution,
                                             self.resolution_time, self.tmax, self.dmax, self.do_tra, self.collision_risk_ob)
                    # print('Ps',Ps)
                    if Ps < 0.2:
                        continue
                    h = self.cost_to_go(s1_state, sG)
                    s1 = Node(s1_state, s1_key, g, g + self.e * h, Ps, sc)
                    if s1 in self.openlist:
                        if s1.cost < self.openlist._heap[self.openlist._idx[s1.key]].cost:
                            self.openlist.update(s1)
                    else:
                        self.openlist.insert(s1)
                        # if fig:
                        #     fig.plot(s1_state[1],s1_state[0],'og',markersize=1)

            if fig:
                try:
                    for plot_line in plot_lines:
                        fig.lines.remove(plot_line[0])
                except BaseException:
                    pass

                try:
                    for plot_patch in plot_patches:
                        fig.patches.remove(plot_patch)
                except BaseException:
                    pass

                plot_lines = []
                plot_patches = []
                # plot_lines.append(fig.plot(sc.state[1], sc.state[0], "or", markersize=5))
                fig.plot(sc.key[1], sc.key[0], "ob", markersize=2)
                fig.add_patch(
                    patches.Arrow(
                        sc.key[1],
                        sc.key[0],
                        0.5 * sin(current_yaw),
                        0.5 * cos(current_yaw),
                        width=0.5))
                if self.do_tra is not None:
                    for id in range(self.do_tra.shape[0]):
                        do_y, do_x, do_yaw = self.do_tra[id, int(sc.state[4]), 1], self.do_tra[id, int(
                            sc.state[4]), 0], self.do_tra[id, int(sc.state[4]), 2]
                        plot_lines.append(
                            fig.plot(
                                do_y,
                                do_x,
                                "or",
                                markersize=5))
                        if int(sc.state[4]) + 10 < self.do_tra.shape[1]:
                            do_y_next, do_x_next = self.do_tra[id, int(
                                sc.state[4]) + 10, 1], self.do_tra[id, int(sc.state[4]) + 10, 0]
                            plot_lines.append(
                                fig.plot([do_y, do_y_next], [do_x, do_x_next], '--r'))
                        i = np.where(self.collision_risk_ob[:, 0] == id)[0]
                        if len(i) > 0:
                            i = i[0]
                            plot_patches.append(plot_colrges_cost_range(
                                do_x, do_y, do_yaw, self.collision_risk_ob[i, 1], fig))
                        plot_lines.append(
                            plot_circle(
                                (do_y, do_x), np.sqrt(
                                    sc.state[4] * C_sigma), fig))
                # if sc.father is not None:
                #     fig.plot([sc.state[1],sc.father.state[1]],[sc.state[0],sc.father.state[0]],"--b")
                plt.pause(0.0001)
        return None

    def state2key(self, s_state):
        x, y, yaw, u, t = s_state
        if dimension==2:
            return (int(round(x / self.resolution_pos) * self.resolution_pos),
                    int(round(y / self.resolution_pos) * self.resolution_pos))
        elif dimension==3:
            return (int(round(x / self.resolution_pos) * self.resolution_pos),
                    int(round(y / self.resolution_pos) * self.resolution_pos),
                    int(round(yaw / pi * 4)))

    def cost_to_go(self, s1_state, sG_state):
        d = np.sqrt((s1_state[0] - sG_state[0])**2 +
                    (s1_state[1] - sG_state[1])**2)
        t = d / self.default_speed
        return Wc * (t / self.tmax) + (1.0 - Wc) * d / self.dmax

    def generate_total_trajectory(self, s):
        trajectory = []
        while s is not None:
            trajectory.append(s.state)
            if s.father is None:
                break
            pos_all = self.compute_trajectory(s.father.state, s.state)
            for i in range(len(pos_all) - 2, -1, -1):
                trajectory.append(pos_all[i])
            s = s.father
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

    def compute_trajectory(self, s_state, s1_state):
        # ucd=self.control_primitives[s_state[3]][(int(s1_state[4]-s_state[4]),"{:.2f}".format(yawRange(s1_state[2]-s_state[2])))]
        ucd = self.control_primitives[s_state[3]][(int(
            s1_state[4] - s_state[4]), np.int(np.round(180 / pi * yawRange(s1_state[2] - s_state[2]))))]
        yaw = s_state[2]
        state_all = []
        for i in range(self.resolution_time - 1, ucd.shape[0], self.resolution_time):
            state_all.append([s_state[0] + ucd[i, 0] * cos(yaw) - ucd[i, 1] * sin(yaw),
                              s_state[1] + ucd[i, 0] * sin(yaw) + ucd[i, 1] * cos(yaw),
                              yawRange(yaw + ucd[i, 2]),
                              ucd[i, 3],
                              s_state[4] + ucd[i, 4]])
        return np.array(state_all)

    def evaluate_encounter(self, sc):
        t = int(sc.state[4])
        if self.do_tra.shape[0] > 0 and self.do_tra.shape[1] > t:
            pos = np.array(sc.state[:2])
            dpos = self.do_tra[:, t, 0:2] - pos
            local_consider_ids = [
                i for i, value in enumerate(dpos) if np.inner(
                    value, value) < self.local_radius**2]
            self.collision_risk_ob = np.zeros(
                (len(local_consider_ids), 3), dtype=np.int)
            for i, id in enumerate(local_consider_ids):
                tcpa, dcpa = get_cpa(sc.state, self.do_tra[id, t])
                encounter_type, times = 0, 0
                if tcpa > 0 and tcpa < self.tcpa_min and dcpa < self.dcpa_min:
                    encounter_type = colrges_encounter_type(
                        sc.state, self.do_tra[id, t])
                if sc.father:
                    id_father = np.where(sc.father.encounter_type[:, 0] == id)[0]
                    if len(id_father) > 0:
                        id_father = id_father[0]
                        if sc.father.encounter_type[id_father, 2] < 5 and sc.father.encounter_type[id_father, 1] > encounter_type:
                            encounter_type = sc.father.encounter_type[id_father, 1]
                            times = sc.father.encounter_type[id_father, 2] + 1
                self.collision_risk_ob[i] = [id, encounter_type, times]
            sc.encounter_type = self.collision_risk_ob
        else:
            self.collision_risk_ob = np.zeros((0, 3), dtype=np.int)


def load_control_primitives(file_path):
    print("load control primitives from {}".format(file_path))
    return np.load(file_path).item()


def plot_circle(c, r, fig):
    theta = np.linspace(0, 2 * np.pi, 800)
    y, x = np.cos(theta) * r + c[0], np.sin(theta) * r + c[1]
    circle = fig.plot(y, x, "--r")
    return circle


def plot_colrges_cost_range(x, y, yaw, encounter_type, fig):
    if encounter_type == 4:
        theta = -yaw * 180 / pi
        wedge = patches.Wedge((y, x), 30, theta, theta + 97.5, color='y')
        return fig.add_patch(wedge)
    elif encounter_type == 3:
        theta = -yaw * 180 / pi + 90
        wedge = patches.Wedge((y, x), 30, theta - 22.5, theta + 45, color='y')
        return fig.add_patch(wedge)


@jit(nopython=True)
def get_cpa(s1, s2):
    x1, y1, yaw1, u1, _ = s1
    x2, y2, yaw2, u2, _ = s2
    dv = np.array([u1 * cos(yaw1) - u2 * cos(yaw2),
                   u1 * sin(yaw1) - u2 * sin(yaw2)])
    dpos = np.array([x1 - x2, y1 - y2])
    fenmu=np.dot(dv, dv)
    if fenmu>0:
        tcpa = -np.dot(dv, dpos) / fenmu
    else:
        tcpa=-np.dot(dv, dpos)*np.inf
    dpos1 = dpos + tcpa * dv
    dcpa = np.sqrt(np.dot(dpos1, dpos1))
    return tcpa, dcpa


# @jit(nopython=True)
# def colrges_encounter_type(s_usv, s_ob):
#     alpha_b = yawRange(
#         np.arctan2(
#             s_usv[1] -
#             s_ob[1],
#             s_usv[0] -
#             s_ob[0]) -
#         s_ob[2])
#     alpha_h = yawRange(s_usv[2] - s_ob[2])
#     if abs(alpha_b) <= pi / 12 and abs(alpha_h) >= 11 * pi / 12:
#         encounter_type = 4
#     elif alpha_b > pi / 12 and alpha_b < 3 * pi / 4 and alpha_h > -11 * pi / 12 and alpha_h < -pi / 4:
#         encounter_type = 1
#     elif alpha_b > -3 * pi / 4 and alpha_b < -pi / 12 and alpha_h > pi / 4 and alpha_h < 11 * pi / 12:
#         encounter_type = 3
#     elif abs(alpha_b) >= 3 * pi / 4 and abs(alpha_h) <= pi / 4:
#         encounter_type = 2
#     else:
#         encounter_type = 0
#     return encounter_type
@jit(nopython=True)
def colrges_encounter_type(s_usv, s_ob):
    alpha_b = yawRange(
        np.arctan2(
            s_usv[1] -
            s_ob[1],
            s_usv[0] -
            s_ob[0]) -
        s_ob[2])
    alpha_h = yawRange(s_usv[2] - s_ob[2])
    if abs(alpha_b) <= pi / 6 and abs(alpha_h) >= 3 * pi / 4:
        encounter_type = 4
    elif alpha_b > pi / 6 and alpha_b < 3 * pi / 4 and alpha_h > -11 * pi / 12 and alpha_h < -pi / 4:
        encounter_type = 0
    elif alpha_b > -3 * pi / 4 and alpha_b < -pi / 6 and alpha_h > pi / 4 and alpha_h < 11 * pi / 12:
        encounter_type = 3
    elif abs(alpha_b) >= 3 * pi / 4 and abs(alpha_h) <= pi / 4:
        encounter_type = 2
    else:
        encounter_type = 1
    return encounter_type

# @jit(nopython=True)
# def extend_new_node_jit(s_state,keys,ucd,s_ps,s_g):
#     gs=np.zeros(len(keys))
#     Pss = np.zeros(len(keys))
#     s1_states=[]
#     for i in prange(len(keys)):
#         s1_state=(s_state[0]+ucd[i,keys[i,0]-1, 0] * cos(s_state[2]) - ucd[i,keys[i,0]-1, 1] * sin(s_state[2]),
#                   s_state[1]+ucd[i,keys[i,0]-1, 0] * sin(s_state[2]) + ucd[i,keys[i,0]-1, 1] * cos(s_state[2]),
#                   yawRange(s_state[2]+np.round(keys[i,1]/15)*15*pi/180), s_state[3], s_state[4] + ucd[i,keys[i,0]-1, 4])
#         g, Ps = cost_to_come(s_state, s1_state, ucd[i],s_ps,s_g)
#         gs[i]=g
#         Pss[i]=Ps
#         s1_states.append(s1_state)
#     return gs,Pss,s1_states


@jit(nopython=True)
def cost_to_come(
        s_state,
        s1_state,
        ucd,
        s_ps,
        s_g,
        static_map,
        map_offset,
        map_resolution,
        resolution_time,
        tmax,
        dmax,
        do_tra,
        collision_risk_ob):
    Pcs, colrges_break = evaluate_primitive(
        s_state, ucd, static_map, map_offset, map_resolution, resolution_time, do_tra, collision_risk_ob)
    # colrges_break=0
    Cs = Wn * (Wc * (s1_state[4] - s_state[4]) / tmax +
               (1.0 - Wc) * ucd[-1, 5] / dmax + C_colrges*colrges_break)
    g = s_g + s_ps / Cg_max * ((1.0 - Pcs) * Cs + Pcs * Ce)
    return g, s_ps * (1 - Pcs)


@jit(nopython=True)
def evaluate_primitive(
        s_state,
        ucd,
        static_map,
        map_offset,
        map_resolution,
        resolution_time,
        do_tra,
        collision_risk_ob):
    primitives = compute_trajectory(s_state, ucd, resolution_time)
    t = np.int(s_state[4])
    colrges_break = 0.0
    no_Pcsu = 1.0
    for i in range(primitives.shape[0]):
        t += resolution_time
        p = collision_with_static_ob(
            primitives[i, 0:2], static_map, map_offset, map_resolution)
        if p == 1.0:
            return 1.0, 0.0
        no_Pcsu_t = 1.0
        if (i + 1) % (primitives.shape[0] // 2) == 0:
            # colrges_cost(primitives[i], np.array([10.0, 10.0, pi, 4.0]), 1)
            # collision_pro_cal(1001.0, C_sigma * t, 1000.0)
            for id, encounter_type in zip(
                    collision_risk_ob[:, 0], collision_risk_ob[:, 1]):
                colrges_break += colrges_cost(
                    primitives[i], do_tra[id, t], encounter_type)
                if encounter_type != 0:
                    # if encounter_type == 4:
                    #     pos_do = do_tra[id, t, 0:2] + 5 * np.array(
                    #         [-sin(do_tra[id, t, 2]), cos(do_tra[id, t, 2])])
                    # elif encounter_type == 3:
                    #     pos_do = do_tra[id, t, 0:2] + 5 * np.array(
                    #         [cos(do_tra[id, t, 2]), sin(do_tra[id, t, 2])])
                    # else:
                    #     pos_do = do_tra[id, t, 0:2]
                    pos_do = do_tra[id, t, 0:2]
                    distance = np.sqrt(
                        np.dot(pos_do - primitives[i, 0:2], pos_do - primitives[i, 0:2]))
                    no_Pcsu_t *= (1 - collision_pro_cal(distance,
                                                        C_sigma * t, 6))
            no_Pcsu *= no_Pcsu_t
    return (1 - no_Pcsu) * exp(-gamma * s_state[4]), colrges_break / 2


@jit(nopython=True)
def compute_trajectory(s_state, ucd, resolution_time):
    yaw = s_state[2]
    state_all = []
    for i in range(resolution_time - 1, ucd.shape[0], resolution_time):
        state_all.append([s_state[0] + ucd[i, 0] * cos(yaw) - ucd[i, 1] * sin(yaw),
                          s_state[1] + ucd[i, 0] * sin(yaw) + ucd[i, 1] * cos(yaw),
                          yawRange(yaw + ucd[i, 2]),
                          ucd[i, 3],
                          s_state[4] + ucd[i, 4]])
    return np.array(state_all)


@jit(nopython=True)
def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


@jit(nopython=True)
def collision_with_static_ob(pos, static_map, map_offset, map_resolution):
    pos_key = (
        np.int(
            np.round(
                (pos[1] - map_offset[0]) / map_resolution)),
        np.int(
            np.round(
                    (pos[0] - map_offset[1]) / map_resolution)))
    if pos_key[0] < 0 or pos_key[0] >= static_map.shape[0] or pos_key[1] < 0 or pos_key[1] >= static_map.shape[1] or static_map[
            pos_key[0], pos_key[1]] == 1:
        return 1.0
    return 0.0


@jit(nopython=True)
def colrges_cost(s_usv, s_ob, encounter_type):
    alpha_b = yawRange(
        np.arctan2(
            s_usv[1] -
            s_ob[1],
            s_usv[0] -
            s_ob[0]) -
        s_ob[2])
    distance = np.sqrt(np.dot(s_usv[0:2] - s_ob[0:2], s_usv[0:2] - s_ob[0:2]))
    if encounter_type == 4 and -pi / 24 < alpha_b < pi / 2 and distance < 30:
        return 1.0
    elif encounter_type == 3 and -pi / 4 < alpha_b < pi / 8 and distance < 30:
        return 1.0
    else:
        return 0.0


@jit(nopython=True)
def collision_pro_cal(d, sigma2, r):
    if d == 0:
        step = r / 10
        x1 = np.arange(0, r, step)
        return np.sum(1 / (2 * pi * sigma2) * exp(-1 / 2 *
                                                  x1 ** 2 / sigma2) * 2 * pi * x1 * step)
    elif d >= r:
        step = (2 * r) / 10
        x = np.arange(d + r - 0.1, d - r, -step)
        return np.sum(np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d) *
                      2 * x * step / (2 * pi * sigma2) * exp(-1 / 2 * x ** 2 / sigma2))
    else:
        step = (2 * d - 0.1) / 10
        step1 = (r - d) / 10
        x = np.arange(d + r - 0.1, r - d, -step)
        x1 = np.arange(0, r - d, step1)
        s = np.sum(np.arccos((x ** 2 + d ** 2 - r ** 2) / 2 / x / d) * 2 * x * step / (2 * pi * sigma2) * exp(-1 / 2 * x ** 2 / sigma2))
        s1 = np.sum(1 / (2 * pi * sigma2) * exp(-1 / 2 * x1 ** 2 / sigma2) * 2 * pi * x1 * step1)
        return s + s1

