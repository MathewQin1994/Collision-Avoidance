import numpy as np
import numpy
import matplotlib.pyplot as plt
import os

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



if __name__=='__main__':
    # filename='../data_record/global_planning/2019-09-23-15-42-26_state.txt'
    # b=np.loadtxt(filename,delimiter=',')
    # fig=plt.gca()
    # fig.plot(b[:,4],b[:,3])
    # plt.show()
    dir_path=r'C:\Users\B209\Desktop\Collision-Avoidance\src\data_record\test_control'
    view_file(dir_path)