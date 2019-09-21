import numpy as np

class DataRecord:
    def __init__(self,filename):
        self.f=open(filename,'w')
    def write(self,data):
        line=','.join([str(i) for i in data])+'\n'
        self.f.write(line)
    def close(self):
        self.f.close()

if __name__=='__main__':
    filename='../data_record/trajectory_follow2019-09-20-14-38-07.txt'
    a=np.loadtxt(filename,delimiter=',')