import numpy as np
import numpy
import matplotlib.pyplot as plt
import os
from collections import deque
from src.tools.msgdev import PeriodTimer
from src.map.staticmap import Map
import matplotlib as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
colors = ['white', 'gold', 'orange', 'blue', 'green', 'purple']
bounds = [0,1,2,3,4,5,6]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

class DataRecord:
    def __init__(self,filename):
        self.f=open(filename,'w')
    def write(self,data):
        if isinstance(data,(list,tuple)):
            line=','.join([str(i) for i in data])+'\n'
            self.f.write(line)
        if isinstance(data,numpy.ndarray):
            shape=data.shape
            if len(shape)==1:
                line = ','.join([str(i) for i in data]) + '\n'
                self.f.write(line)
            elif len(shape)==2:
                for i in range(shape[0]):
                    line = ','.join([str(i) for i in data[i,:]]) + '\n'
                    self.f.write(line)
            elif len(shape)==3:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        line = ','.join([str(i) for i in data[i,j,:]]) + '\n'
                        self.f.write(line)
    def close(self):
        self.f.close()

def view_file(dir_path):
    file_list=os.listdir(dir_path)
    for filename in file_list:
        dyaw=int(filename.split('_')[-1].split('.')[0])
        print(dyaw)
        data=np.loadtxt(os.path.join(dir_path,filename),delimiter=',')
        fig=plt.figure()
        plt.plot(data[:, 4], data[:, 3])
        plt.axis('equal')
        plt.pause(0.1)

class DataReappear:
    def __init__(self,dir_path,date,case):
        self.do_file=open(os.path.join(dir_path,date+'_do.txt'),'r')
        self.point_file = open(os.path.join(dir_path, date + '_point.txt'),'r')
        self.state_file = open(os.path.join(dir_path, date + '_state.txt'),'r')
        self.x_q = deque()
        self.y_q = deque()
        self.dt=1
        self.dT=6
        self.case=case

    def load_map(self,case):
        static_map = Map()
        if case == 'case1':
            static_map.load_map(np.loadtxt('../map/static_map1.txt', dtype=np.int8), resolution=0.5)
        elif case == 'case2':
            static_map.load_map(np.loadtxt('../map/static_map2.txt', dtype=np.int8), resolution=0.5)
        elif case == 'case3':
            static_map.load_map(np.loadtxt('../map/static_map3.txt', dtype=np.int8), resolution=1, offset=(-80, -35))
        elif case == 'case4':
            static_map.load_map(np.loadtxt('../map/static_map4.txt', dtype=np.int8), resolution=1, offset=(-80, -35))
        elif case == 'case5':
            static_map.load_map(np.loadtxt('../map/static_map5.txt', dtype=np.int8), resolution=1, offset=(-80, -35))
        elif case == 'case0':
            static_map.new_map(size=(100, 100), offset=(-63, -54))
        else:
            raise Exception
        fig = plt.gca()
        extend = [
            static_map.offset[0],
            static_map.size[0] *
            static_map.resolution +
            static_map.offset[0] - 1,
            static_map.offset[1],
            static_map.size[1] *
            static_map.resolution +
            static_map.offset[1] - 1]
        mapplot = static_map.map.copy()
        for i in range(mapplot.shape[0]):
            mapplot[i, :] = mapplot[i, :][::-1]
        fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
        fig.set_xlabel('E/m')
        fig.set_ylabel('N/m')
        fig.axis("equal")
        return fig

    def reappear(self):
        fig=self.load_map(self.case)
        plot_lines_state=[]
        plot_lines_tra=[]
        plot_lines_do_tra = []
        self.do_tra=self.read_do()
        self.target_points=self.read_point()
        t=PeriodTimer(0.1)
        t.start()
        while True:
            with t:
                self.update_state(fig,plot_lines_state)
                self.update_planning(fig,plot_lines_tra)
                self.update_do_pre(fig,plot_lines_do_tra)
                fig.legend(loc='upper left')
                a=fig.text(65,70,'T={}s'.format(t.i*self.dt))
                plt.pause(0.01)
                a.set_visible(False)


    def read_do(self):
        head=self.do_file.readline().rstrip('\n').split(',')
        if head[0]=='':
            return np.zeros((0,0,5))
        data=[]
        for _1 in range(int(head[1])):
            data1=[]
            for _2 in range(int(head[2])):
                data1.append([float(i) for i in self.do_file.readline().rstrip('\n').split(',')])
            data.append(data1)
        return np.array(data)

    def read_point(self):
        head=self.point_file.readline().rstrip('\n').split(',')
        if head[0]=='':
            return np.zeros((0,5))
        data=[]
        for _ in range(int(head[1])):
            data.append([float(i) for i in self.point_file.readline().rstrip('\n').split(',')])
        return np.array(data)

    def close(self):
        self.do_file.close()
        self.point_file.close()
        self.state_file.close()

    def update_state(self, fig, plot_lines):
        s_ob = [float(i) for i in self.state_file.readline().rstrip('\n').split(',')]
        self.time=s_ob[-1]
        self.x_q.append(s_ob[3])
        self.y_q.append(s_ob[4])
        if len(self.x_q) > 60 / self.dt:
            self.x_q.popleft()
            self.y_q.popleft()
        while len(plot_lines) > 0:
            fig.lines.remove(plot_lines.pop()[0])
        plot_lines.append(fig.plot(self.y_q, self.x_q, 'b',label='本船历史轨迹'))
        plot_lines.append(fig.plot(s_ob[4], s_ob[3], 'b*',markersize=8,label='本船当前位置'))

    def update_planning(self, fig, plot_lines):
        if self.target_points.shape[0]==0:
            while len(plot_lines) > 0:
                fig.lines.remove(plot_lines.pop()[0])
            return
        if self.time>=self.target_points[0,-1]:
            while len(plot_lines) > 0:
                fig.lines.remove(plot_lines.pop()[0])
            # for i in range(self.target_points.shape[0]):
            plot_lines.append(fig.plot(self.target_points[:, 1], self.target_points[:, 0], 'bo', markersize=2,label='本船规划轨迹点'))
            self.target_points=self.read_point()

    def update_do_pre(self, fig, plot_lines):
        if self.do_tra.shape[0]==0:
            while len(plot_lines) > 0:
                fig.lines.remove(plot_lines.pop()[0])
            return
        if self.time>=self.do_tra[0,0,-1] and self.time<self.do_tra[0,0,-1]+self.dT:
            while len(plot_lines) > 0:
                fig.lines.remove(plot_lines.pop()[0])
            j=int(self.time-self.do_tra[0,0,-1])
            for i in range(self.do_tra.shape[0]):
                plot_lines.append(fig.plot(self.do_tra[i, j:, 1], self.do_tra[i, j:, 0], 'r--',label='他船预测轨迹'))
                plot_lines.append(fig.plot(self.do_tra[i, j, 1], self.do_tra[i, j, 0], '*r', markersize=8,label='他船当前位置'))
        elif self.time>=self.do_tra[0,0,-1]+self.dT:
            self.do_tra=self.read_do()


if __name__=='__main__':
    # filename='../data_record/global_planning/2019-09-23-15-42-26_state.txt'
    # b=np.loadtxt(filename,delimiter=',')
    # fig=plt.gca()
    # fig.plot(b[:,4],b[:,3])
    # plt.show()
    dir_path=r'C:\Users\40350\Desktop\研二上\毕业论文\Collision Avoidance\src\data_record\global_planning'
    date='2019-09-26-15-45-17'
    # date='2019-09-26-13-12-28'
    da=DataReappear(dir_path,date,'case5')
    try:
        da.reappear()
    except:
        da.close()
        raise
    # view_file(dir_path)