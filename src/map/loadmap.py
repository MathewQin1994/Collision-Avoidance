import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from src.map.staticmap import Map, generate_do_trajectory
import matplotlib as mpl

def rotato_right_90(matrix):
    tmp=matrix.T
    i,j=0,tmp.shape[1]-1
    while i<j:
        for k in range(tmp.shape[0]):
            tmp[k,i],tmp[k,j]=tmp[k,j],tmp[k,i]
        i+=1
        j-=1
    return tmp



# 图片二值化

from PIL import Image
Img = Image.open('map.png')
#图片实际高度

# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = Img.rotate(0)
Img = Img.convert('L')
# Img.save("test1.jpg")
height=100
imgsize=Img.size
ratio=Img.size[1]//height
img1=Img.resize((imgsize[0]//ratio,height))

#自定义灰度界限，大于这个值为黑色，小于这个值为白色
water_range=(209,211)

table = []
for i in range(256):
    if water_range[0]<=i<=water_range[1]:
        table.append(0)
    else:
        table.append(1)

# 图片二值化
a = np.asanyarray(img1.point(table, '1'),dtype=np.int8)
a=rotato_right_90(a)

# photo.save("test2.jpg")

# b=np.loadtxt('../data_record/global_planning/2019-09-23-13-25-15_state.txt',delimiter=',')
colors = ['white', 'gold', 'orange', 'blue', 'green', 'purple']
bounds = [0,1,2,3,4,5,6]

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

static_map = Map()
# static_map.load_map(np.loadtxt('../map/static_map1.txt',dtype=np.int8), 1)
static_map.load_map(a, 1,offset=(-80,-35))

#静态障碍物
circle_static_obstacles=((36,14,8),(33,-6,8))
for ob in circle_static_obstacles:
    static_map.add_static_obstacle(type="circle", config=ob)
np.savetxt('static_map5.txt',static_map.map)

fig = plt.gca()
extend = [
    static_map.offset[0],
    static_map.size[0] *
    static_map.resolution +
    static_map.offset[0],
    static_map.offset[1],
    static_map.size[1] *
    static_map.resolution +
    static_map.offset[1]]
fig.set_xlabel('E/m')
fig.set_ylabel('N/m')
mapplot = static_map.map.copy()
for i in range(mapplot.shape[0]):
    mapplot[i, :] = mapplot[i, :][::-1]
fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
# fig.plot(b[:,4],b[:,3])
plt.show()
