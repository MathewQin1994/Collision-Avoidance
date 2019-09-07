#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
# import rospy
# from sailboat_message.msg import Ahrs_msg
# from sailboat_message.msg import WTST_msg
# from sailboat_message.msg import Sensor_msg
import pyqtgraph as pg

app = pg.mkQApp()

win = pg.GraphicsWindow()
win.setWindowTitle(u'Sensor Monitor')
win.resize(800, 800)


p1 = win.addPlot(title='Yaw',labels={'left':'yaw/rad','bottom':'t/s'})
p1.showGrid(x=True, y=True)
curve1 = p1.plot(pen=(255, 0, 0), name="Red curve")
curve2 = p1.plot(pen=(0, 255, 0), name="Green curve")
curve3 = p1.plot(pen=(0, 0, 255), name="Blue curve")
i=0
data=[]
x=[]
starttime=time.time()
def update():
    global i
    tmp=[AHRSdata[i,3],WTSTdata[i,7],WTSTdata[i,3]]
    # tmp=AHRSdata[i,3]
    if len(data)<200:
        data.append(tmp)
        x.append(i)
    else:
        data[:-1] = data[1:]
        data[-1] = tmp
        x[:-1]=x[1:]
        x[-1]=i
    array=np.array(data)
    print(time.time()-starttime)
    curve1.setData(x,array[:,0])
    curve2.setData(x, array[:, 1])
    curve3.setData(x, array[:, 2])
    # p1.plot().setData(x,array[:,1])
    # p1.plot().setData(x,array[:,2])
    i+=1
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000)

# if __name__ == "__main__":
app.exec_()