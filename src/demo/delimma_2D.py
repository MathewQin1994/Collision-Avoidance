from src.planner.Astar3D_jit import DeliberativePlanner
from numpy import pi
from src.map.staticmap import Map, generate_do_trajectory
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

def main():
    # 参数
    resolution_time = 1
    resolution_pos = 1
    default_speed = 0.8
    primitive_file_path = '../primitive/control_primitives.npy'
    e = 1.2

    # 地图、起点、目标
    map_size = (100, 100)
    static_map = Map()
    static_map.new_map(map_size, resolution=1)
    fig = plt.gca()
    fig.axis([0, map_size[0], 0, map_size[1]])
    fig.set_xlabel('E/m')
    fig.set_ylabel('N/m')
    s0 = tuple(np.array((15, 49, 0, 0.8, 0), dtype=np.float64))
    sG = tuple(np.array((80, 48, pi, 0.8, 0), dtype=np.float64))
    fig.plot(sG[1], sG[0], "ob", markersize=5)

    # 静态障碍物
    # rectangle_static_obstacles = ((10, 50, 50, 10), (50, 10, 10, 40))
    # rectangle_static_obstacles = ((0, 20, 80, 10), (20, 50, 80, 10))
    # rectangle_static_obstacles = ((40, 30, 60, 10), (20, 60, 20, 20))
    # rectangle_static_obstacles=((40,75,70,50),(150,75,100,50),(175,175,60,60))
    rectangle_static_obstacles = ((0, 50, 60, 10), (50, 0, 10, 48))
    # rectangle_static_obstacles = ()

    for ob in rectangle_static_obstacles:
        static_map.add_static_obstacle(type="rectangle", config=ob)
        rect = patches.Rectangle((ob[0], ob[1]), ob[2], ob[3], color='y')
        fig.add_patch(rect)

    # 动态障碍物
    # do_tra = np.array([generate_do_trajectory(95, 100, -3 * pi / 4, 0.70, 200)])

    dp = DeliberativePlanner(
        static_map,
        resolution_pos,
        resolution_time,
        default_speed,
        primitive_file_path,
        e)
    start_time = time.time()
    # dp.set_dynamic_obstacle(do_tra)
    tra = np.array(dp.start(s0, sG))
    logging.info("runtime is {},closelist node number is {},trajectory total time is {}".format(
        time.time() - start_time, len(dp.closelist), tra[-1, -1]))
    # for i in range(do_tra.shape[0]):
    #     fig.plot(do_tra[i, :, 1], do_tra[i, :, 0], "r")
    fig.plot(tra[:, 1], tra[:, 0], "b")
    # for i in range(tra.shape[0]):
    #     if tra[i, 3] == 0.8:
    #         fig.plot(tra[i, 1], tra[i, 0], "or", markersize=2)

    # a = np.array(list(dp.closelist), dtype=np.float64)
    # fig.plot(a[:,1],a[:,0],'ob',markersize=1)

    plt.show()

if __name__=="__main__":
    main()