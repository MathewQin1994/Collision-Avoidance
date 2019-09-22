import numpy as np
import numpy

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

if __name__=='__main__':
    filename='../data_record/trajectory_follow2019-09-20-14-38-07.txt'
    a=np.loadtxt(filename,delimiter=',')