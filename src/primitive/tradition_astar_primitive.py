import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.pyplot as plt
from src.primitive.Trimaran import control_primitives_visual

def get_all_control_primitives(save=True):
    # time_set=np.array([10,5],dtype=np.int)
    u=0.8
    control_primitives=dict()
    action_time=1
    control_primitives[u]=dict()
    yaw_set = np.array([-3*pi / 4, -pi / 2,-pi/4, 0, pi/4,pi / 2, 3*pi / 4,pi], dtype=np.float64)
    for yaw in yaw_set:
        key=(action_time,np.int(np.round(yaw*180/pi)))
        control_primitives[u][key]=control_action_primitives(u,yaw,action_time)
    if save:
        np.save('tradition_astar_primitives{}.npy'.format(u),control_primitives)
    return control_primitives

def control_action_primitives(u,yaw,action_time):
    control_primitives=np.array([[u*action_time*cos(yaw),u*action_time*sin(yaw),yaw,u,action_time,u*action_time]])
    return control_primitives



if __name__=="__main__":
    control_primitives=get_all_control_primitives(save=True)
    control_primitives_visual(control_primitives)