# Collision-Avoidance
基于A*搜索算法的考虑实时性约束、船舶操纵性约束、海事规则约束的慎思避碰规划算法


# Simulation
1.deliberative planning

地图中，紫色区域为海域，黄色区域为静态障碍物，蓝色实线代表本船历史轨迹，蓝色虚线代表本船规划轨迹，红色实线代表他船历史轨迹，红色虚线代表他船预测轨迹，
重规划频率为0.1Hz；

case1：不考虑COLREGs规则的全局规划
![gif](https://github.com/MathewQin1994/Collision-Avoidance/tree/master/gif/case2.1.gif)

case2：考虑COLREGs规则的全局规划
![gif](https://github.com/MathewQin1994/Collision-Avoidance/tree/master/gif/case2.2.gif)

case3：考虑COLREGs规则的多船复杂会遇情况的全局规划
![gif](https://github.com/MathewQin1994/Collision-Avoidance/tree/master/gif/case1.gif)

case4：考虑COLREGs规则的多船复杂会遇情况的全局规划，相比case3加大了与他船碰撞cost
![gif](https://github.com/MathewQin1994/Collision-Avoidance/tree/master/gif/case1.1.gif)

2.trajectory following

![gif](https://github.com/MathewQin1994/Collision-Avoidance/tree/master/gif/tra_follow.gif)
