# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:03:45 2022

@author: mirko
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib import path
from copy import deepcopy
import time
from gen_live import create_geo
import subprocess
import os, sys
import shutil
sys.path.insert(1, '/home/mirko/')
from model import predict

show_vectors = False
if len(sys.argv) > 1:
    if str(sys.argv[1]).lower() == 'vectors':
        show_vectors = True



sim_started = 0
step_completed = 0
finished = 0
process = 0
get_latest = 0
last_xo = []
last_yo = []
last_yin = []


geom = np.ones((85,256))
geom[:35,:43] = 0
geom[-35:,:43] = 0

X, Y = np.meshgrid(np.arange(256), np.arange(85))

xs = np.arange(1,256,8)
ys = np.arange(1,85,4)
Xs, Ys = np.meshgrid(xs,ys)



def start_sim(yin, xo, yo, geom):
    global sim_started
    global process
    global get_latest
    global finished
    global fig, ax4, img4, quiv4
    global Xs, Ys

    finished = 0
    # if len(xo) == 0:
    #     return
    if len(yin) == 0:
        yin = [34, 49]
    for i in range(len(yin)):
        yin[i] = yin[i]/85
    xo = xo[:-1]
    yo = yo[:-1]
    for i in range(len(xo)):
        xo[i] = xo[i]/256*3
    for i in range(len(yo)):
        yo[i] = yo[i]/85
    create_geo(yin, xo, yo)
    if process != 0:
        sim_started = 0
        ax4.set_title('simulation (terminating)')
        img4.set_data(geom*0)
        quiv4.set_UVC(Xs*0,-Xs*0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        print('Terminating!')
        # while not isinstance(get_latest,subprocess.CompletedProcess):
        #     print('move on')
        #     pass
        with open('simfiles/live_sim/system/controlDict', 'r') as file :
            controlDict = file.read()
        controlDict = controlDict.replace('endTime;', 'noWriteNow;')
        with open('simfiles/live_sim/system/controlDict', 'w') as file:
            while not os.path.isfile('simfiles/live_sim/done.btch'):
                file.write(controlDict)
                time.sleep(0.01)
        # for i in range(1000):
            # os.popen('cp simfiles/live_sim/system/alt_cntrl simfiles/live_sim/system/controlDict') 
            # shutil.copyfile('simfiles/live_sim/system/alt_cntrl', 'simfiles/live_sim/system/controlDict')
        # os.remove("simfiles/base_case/system/controlDict") 
        
    try:
        shutil.rmtree("./simfiles/live_sim")
    except:
        pass
    sim_started = 1
    process = subprocess.Popen("./study.sh", cwd='simfiles')
    

def update_obstacle(xs, ys, geom, predict):
    # print(xs)
    # geom[:,43:] = 1
    vertices = []
    for i in range(len(xs)):
        vertices.append((xs[i],ys[i]))
    obstacle = path.Path(vertices)
    flags = obstacle.contains_points(np.hstack((X.flatten()[:,np.newaxis],Y.flatten()[:,np.newaxis])))
    flags = np.reshape(flags, np.shape(geom))
    geom = (1-flags)*geom
    prediction = np.squeeze(predict(np.expand_dims(geom,[0,3])))
    Ux = prediction[:,:,0]
    Uy = prediction[:,:,1]
    p  = prediction[:,:,2]
    Umag = np.linalg.norm(prediction[:,:,:2], axis=2)
    return geom, Umag, Ux, Uy, p


def update_inlet(ys, geom, predict):
    geom[:,:43] = 0
    geom[int(min(ys)):int(max(ys)),:43] = 1
    prediction = np.squeeze(predict(np.expand_dims(geom,[0,3])))
    Ux = prediction[:,:,0]
    Uy = prediction[:,:,1]
    p  = prediction[:,:,2]
    Umag = np.linalg.norm(prediction[:,:,:2], axis=2)
    return geom, Umag, Ux, Uy, p


class build_geom:
    def __init__(self, oline, inlet, geom):
        self.oline = oline
        self.inlet = inlet
        self.xo = list(oline.get_xdata())
        self.yo = list(oline.get_ydata())
        self.xin = list(inlet.get_xdata())
        self.yin = list(inlet.get_ydata())
        self.geom = geom
        self.cid = oline.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        global last_xo
        global last_yo
        global last_yin
        # print('click', event)
        if event.inaxes!=self.oline.axes: return
        if event.xdata < 43:
            if len(self.xin) == 0:
                self.xin.append(event.xdata)
                self.yin.append(event.ydata)
            elif len(self.xin) == 1:
                self.xin.append(event.xdata)
                self.yin.append(event.ydata)
                last_yin = self.yin
                self.geom, Umag2, Ux2, Uy2 = update_inlet(self.yin,self.geom, predict_pureGAN)[:4]
                img1.set_data(self.geom)
                img2.set_data(Umag2)
                quiv2.set_UVC(Ux2[Ys,Xs],-Uy2[Ys,Xs])
                img2.set_clim(np.amin(Umag2),np.amax(Umag2))
                Umag3, Ux3, Uy3 = update_inlet(self.yin, self.geom, predict_pureL1)[1:4]
                img3.set_data(Umag3)
                img3.set_clim(np.amin(Umag3),np.amax(Umag3))
                quiv3.set_UVC(Ux3[Ys,Xs],-Uy3[Ys,Xs])
                Umag5, Ux5, Uy5 = update_inlet(self.yin, self.geom, predict_mixed)[1:4]
                img5.set_data(Umag5)
                img5.set_clim(np.amin(Umag5),np.amax(Umag5))
                quiv5.set_UVC(Ux5[Ys,Xs],-Uy5[Ys,Xs])
                start_sim(deepcopy(last_yin), deepcopy(last_xo), deepcopy(last_yo), self.geom)
                img4.set_data(self.geom)
                quiv4.set_UVC(Ux3[Ys,Xs]*0,-Uy3[Ys,Xs]*0)
                img4.set_clim(0,1)
                ax4.set_title('simulation (running)')
                ax2.set_title('adversarially trained')
                ax3.set_title('L1 trained')
                ax5.set_title('combined training')
                self.xin = []
                self.yin = []
                
            self.inlet.set_data(self.xin, self.yin)
            self.inlet.figure.canvas.draw()

        else:
            if event.button is MouseButton.LEFT:
                if len(self.xo) > 0:
                    if self.xo[0] == self.xo[-1] and len(self.xo)>1:
                        self.xo = []
                        self.yo = []
                ynew = event.ydata
                if ynew > 83.5:
                    ynew = 83.5
                elif ynew < 0.5:
                    ynew = 0.5
                self.xo.append(event.xdata)
                self.yo.append(ynew)
                # print(event.ydata)

            if event.button is MouseButton.RIGHT:
                if len(self.xo) > 0:
                    self.xo.append(self.xo[0])
                    self.yo.append(self.yo[0])
                    last_xo = self.xo
                    last_yo = self.yo
                    self.geom, Umag2, Ux2, Uy2 = update_obstacle(self.xo,self.yo, self.geom, predict_pureGAN)[:4]
                    img1.set_data(self.geom)
                    img2.set_data(Umag2)
                    quiv2.set_UVC(Ux2[Ys,Xs],-Uy2[Ys,Xs])
                    img2.set_clim(np.amin(Umag2),np.amax(Umag2))
                    Umag3, Ux3, Uy3 = update_obstacle(self.xo,self.yo, deepcopy(self.geom), predict_pureL1)[1:4]
                    img3.set_data(Umag3)
                    img3.set_clim(np.amin(Umag3),np.amax(Umag3))
                    quiv3.set_UVC(Ux3[Ys,Xs],-Uy3[Ys,Xs])
                    Umag5, Ux5, Uy5 = update_obstacle(self.xo,self.yo, deepcopy(self.geom), predict_mixed)[1:4]
                    img5.set_data(Umag5)
                    img5.set_clim(np.amin(Umag5),np.amax(Umag5))
                    quiv5.set_UVC(Ux5[Ys,Xs],-Uy5[Ys,Xs])
                    start_sim(deepcopy(last_yin), deepcopy(last_xo), deepcopy(last_yo), self.geom)
                    img4.set_data(self.geom)
                    quiv4.set_UVC(Ux3[Ys,Xs]*0,-Uy3[Ys,Xs]*0)
                    img4.set_clim(0,1)
                    ax4.set_title('simulation (running)')
                    ax2.set_title('adversarially trained')
                    ax3.set_title('L1 trained')
                    ax5.set_title('combined training')
                    self.xo = []
                    self.yo = []

            self.oline.set_data(self.xo, self.yo)
            self.oline.figure.canvas.draw()

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(511)
img1 = ax1.imshow(geom)
ax1.set_title('geometry')
ax1.set_axis_off()
bar1 = plt.colorbar(img1, ax=ax1, shrink=0, ticks=[])

ax2 = fig.add_subplot(513)
img2 = ax2.imshow(geom*0)
quiv2 = ax2.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv2.set_visible(show_vectors)
img2.set_clim(0,1)
ax2.set_title('adversarially trained')
ax2.set_axis_off()
bar2 = plt.colorbar(img2, ax=ax2, label='U')

ax3 = fig.add_subplot(512)
img3 = ax3.imshow(geom*0)
quiv3 = ax3.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv3.set_visible(show_vectors)
img3.set_clim(0,1)
ax3.set_title('L1 trained')
ax3.set_axis_off()
bar3 = plt.colorbar(img3, ax=ax3, label='U')

ax4 = fig.add_subplot(515)
img4 = ax4.imshow(geom*0)
quiv4 = ax4.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv4.set_visible(show_vectors)
img4.set_clim(0,1)
ax4.set_title('simulation')
ax4.set_axis_off()
bar4 = plt.colorbar(img4, ax=ax4, label='U')

ax5 = fig.add_subplot(514)
img5 = ax5.imshow(geom*0)
quiv5 = ax5.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv5.set_visible(show_vectors)
img5.set_clim(0,1)
ax5.set_title('combined training')
ax5.set_axis_off()
bar5 = plt.colorbar(img5, ax=ax5, label='U')

oline, = ax1.plot([], [], marker = 'o', color = 'k')  # empty line
inlet, = ax1.plot([], [], marker = 'o', c = 'r', lw=0)  # empty line
live_edit = build_geom(oline,inlet,geom)

# plt.show()

while True:
    # print('NOW')
    if sim_started == 1 and finished != 2:
        # print('sim_started:')
        # print(sim_started)
        if '100' in os.listdir("simfiles/live_sim/"):
            # print('getLatest')
            get_latest = subprocess.run("./get_latest.sh", cwd='simfiles')
            sim = np.load('simfiles/live_sim/sim_result.npy')
            Usim = np.linalg.norm(sim[:,:,1:3], axis=2)
            img4.set_data(Usim)
            quiv4.set_UVC(sim[Ys,Xs,1],-sim[Ys,Xs,2])
            img4.set_clim(np.amin(Usim),np.amax(Usim))
            if finished == 1:
                finished = 2
                simimg = img4.get_array()
                GANimg = img2.get_array()
                L1img = img3.get_array()
                mixedimg = img5.get_array()
                MAE_GAN = round(np.mean(np.abs(GANimg-simimg)),2)
                MAE_L1 = round(np.mean(np.abs(L1img-simimg)),2)
                MAE_mixed = round(np.mean(np.abs(mixedimg-simimg)),2)
                ax4.set_title('simulation (converged)')
                ax2.set_title('adversarially trained, MAE: %.2f' % MAE_GAN)
                ax3.set_title('L1 trained, MAE: %.2f' % MAE_L1)
                ax5.set_title('combined training, MAE: %.2f' % MAE_mixed)
            elif process.poll() != None:
                finished = 1

    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    time.sleep(0.1)