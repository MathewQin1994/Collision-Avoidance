import numpy as np
from numpy import sin,cos,pi,ceil
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from src.map.staticmap import Map, generate_do_trajectory
import matplotlib as mpl

# 图片二值化

from PIL import Image
img = Image.open('map.png')
#图片实际高度

# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')
# Img.save("test1.jpg")
height=100
imgsize=Img.size
ratio=Img.size[1]/height
img1=Img.resize((imgsize[0]/ratio,height))

# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
# threshold = 200
#
# table = []
# for i in range(256):
#     if i < threshold:
#         table.append(0)
#     else:
#         table.append(1)
#
# # 图片二值化
# photo = Img.point(table, '1')
# photo.save("test2.jpg")


# colors = ['white', 'green', 'orange', 'blue', 'yellow', 'purple']
# bounds = [0,1,2,3,4,5,6]
#
# cmap = mpl.colors.ListedColormap(colors)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
# # plt.imshow(my_array, interpolation='none', cmap=cmap, norm=norm)
# static_map = Map()
# static_map.load_map(np.loadtxt('../map/static_map1.txt',dtype=np.int8), 1)
#
# fig = plt.gca()
# extend = [
#     static_map.offset[0],
#     static_map.size[0] *
#     static_map.resolution +
#     static_map.offset[0],
#     static_map.offset[1],
#     static_map.size[1] *
#     static_map.resolution +
#     static_map.offset[1]]
# fig.set_xlabel('E/m')
# fig.set_ylabel('N/m')
# mapplot = static_map.map.copy()
# for i in range(mapplot.shape[0]):
#     mapplot[i, :] = mapplot[i, :][::-1]
# fig.imshow(mapplot.T, extent=extend, interpolation='none', cmap=cmap, norm=norm)
# plt.show()
