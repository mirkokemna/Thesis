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
from PIL import Image
from model import predict

show_vectors = False
if len(sys.argv) > 1:
    if str(sys.argv[1]).lower() == 'vectors':
        show_vectors = True


model_A = 'GAN_baseline'
model_B = 'mixed_1GAN_10L1'
model_C = 'L1'

steps = 50000

third_plot = True

prediction_A = np.zeros((85,256,2))
prediction_B = np.zeros((85,256,2))
prediction_C = np.zeros((85,256,2))

Umax = 1
Umax_sim = 1

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

def on_press(event):
    # print('press', event.key)
    # sys.stdout.flush()
    if event.key == ' ':
        quiv2.set_visible(not quiv2.get_visible())
        quiv3.set_visible(not quiv3.get_visible())
        quiv4.set_visible(not quiv4.get_visible())
        quiverr2.set_visible(not quiverr2.get_visible())
        quiverr3.set_visible(not quiverr3.get_visible())
        if third_plot:
            quiv5.set_visible(not quiv5.get_visible())
            quiverr5.set_visible(not quiverr5.get_visible())
    #     visible = xl.get_visible()
    #     xl.set_visible(not visible)
    #     fig.canvas.draw()

def start_sim(yin, xo, yo, geom):
    global sim_started
    global process
    global get_latest
    global finished
    global fig, ax4, img4, quiv4
    global Xs, Ys

    axerr2.set_title('')
    axerr3.set_title('')
    if third_plot:
        axerr5.set_title('')

    finished = 0
    # if len(xo) == 0:
    #     return
    if len(yin) == 0:
        yin = [35, 50]
    for i in range(len(yin)):
        yin[i] = yin[i]/85
    print(yin)
    xo = xo[:-1]
    yo = yo[:-1]
    for i in range(len(xo)):
        xo[i] = xo[i]/256*3
    for i in range(len(yo)):
        yo[i] = yo[i]/85
    create_geo(yin, xo, yo)
    print(yin)
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
    

def update_obstacle(xs, ys, geom, model):
    # print(xs)
    geom[:,43:] = 1
    vertices = []
    for i in range(len(xs)):
        vertices.append((xs[i],ys[i]))
    obstacle = path.Path(vertices)
    flags = obstacle.contains_points(np.hstack((X.flatten()[:,np.newaxis],Y.flatten()[:,np.newaxis])))
    flags = np.reshape(flags, np.shape(geom))
    geom = (1-flags)*geom
    prediction = np.squeeze(predict(geom,model,steps))
    Ux = prediction[:,:,0]
    Uy = prediction[:,:,1]
    # p  = prediction[:,:,2]
    Umag = np.linalg.norm(prediction[:,:,:2], axis=2)
    return geom, Umag, Ux, Uy#, p


def update_inlet(ys, geom, model):
    geom[:,:43] = 0
    geom[int(min(ys)):int(max(ys)),:43] = 1
    prediction = np.squeeze(predict(geom,model,steps))
    Ux = prediction[:,:,0]
    Uy = prediction[:,:,1]
    # p  = prediction[:,:,2]
    Umag = np.linalg.norm(prediction[:,:,:2], axis=2)
    return geom, Umag, Ux, Uy#, p


class update_plots:
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
        global Umax
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
                self.geom, Umag2, Ux2, Uy2 = update_inlet(self.yin,self.geom, model_A)[:4]
                prediction_A[...,0] = Ux2
                prediction_A[...,1] = Uy2
                img1.set_data(self.geom)
                img2.set_data(Umag2)
                quiv2.set_UVC(Ux2[Ys,Xs],-Uy2[Ys,Xs])
                
                Umag3, Ux3, Uy3 = update_inlet(self.yin, self.geom, model_B)[1:4]
                prediction_B[...,0] = Ux3
                prediction_B[...,1] = Uy3
                img3.set_data(Umag3)
                
                quiv3.set_UVC(Ux3[Ys,Xs],-Uy3[Ys,Xs])
                if third_plot:
                    Umag5, Ux5, Uy5 = update_inlet(self.yin, self.geom, model_C)[1:4]
                    prediction_C[...,0] = Ux5
                    prediction_C[...,1] = Uy5
                    img5.set_data(Umag5)
                    quiv5.set_UVC(Ux5[Ys,Xs],-Uy5[Ys,Xs])

                    Umax = max(np.amax(Umag2),np.amax(Umag3),np.amax(Umag5))
                    img2.set_clim(0,Umax)
                    img3.set_clim(0,Umax)
                    img5.set_clim(0,Umax)

                else:
                    Umax = max(np.amax(Umag2),np.amax(Umag3))
                    img2.set_clim(0,Umax)
                    img3.set_clim(0,Umax)                    

                start_sim(deepcopy(last_yin), deepcopy(last_xo), deepcopy(last_yo), self.geom)
                img4.set_data(self.geom)
                quiv4.set_UVC(Ux3[Ys,Xs]*0,-Uy3[Ys,Xs]*0)
                img4.set_clim(0,1)
                ax4.set_title('simulation (running)')
                ax2.set_title('GAN model')
                ax3.set_title('L1 model')
                if third_plot:
                    ax5.set_title('hybrid model\nweighting: 1xGAN, 10xL1')
                self.xin = []
                self.yin = []
                
            self.inlet.set_data(self.xin, self.yin)
            self.inlet.figure.canvas.draw()

        else:
            # print(event.xdata)
            # print(event.ydata)
            if event.button is MouseButton.LEFT:
                if len(self.xo) > 0:
                    if self.xo[0] == self.xo[-1] and self.yo[0] == self.yo[-1] and len(self.xo)>1:
                        # print('here')
                        # print(self.xo)
                        self.xo = []
                        self.yo = []
                ynew = event.ydata
                if ynew > 83.5:
                    ynew = 83.5
                elif ynew < 0.5:
                    ynew = 0.5
                self.xo.append(event.xdata)
                self.yo.append(ynew)
                

            if event.button is MouseButton.RIGHT:
                if len(self.xo) > 0:
                    self.xo.append(self.xo[0])
                    self.yo.append(self.yo[0])
                    last_xo = self.xo
                    last_yo = self.yo
                    self.geom, Umag2, Ux2, Uy2 = update_obstacle(self.xo,self.yo, self.geom, model_A)[:4]
                    prediction_A[...,0] = Ux2
                    prediction_A[...,1] = Uy2
                    img1.set_data(self.geom)
                    img2.set_data(Umag2)
                    quiv2.set_UVC(Ux2[Ys,Xs],-Uy2[Ys,Xs])
                    Umag3, Ux3, Uy3 = update_obstacle(self.xo,self.yo, deepcopy(self.geom), model_B)[1:4]
                    prediction_B[...,0] = Ux3
                    prediction_B[...,1] = Uy3
                    img3.set_data(Umag3)
                    quiv3.set_UVC(Ux3[Ys,Xs],-Uy3[Ys,Xs])
                    if third_plot:
                        Umag5, Ux5, Uy5 = update_obstacle(self.xo,self.yo, deepcopy(self.geom), model_C)[1:4]
                        prediction_C[...,0] = Ux5
                        prediction_C[...,1] = Uy5
                        img5.set_data(Umag5)
                        quiv5.set_UVC(Ux5[Ys,Xs],-Uy5[Ys,Xs])

                        Umax = max(np.amax(Umag2),np.amax(Umag3),np.amax(Umag5))
                        img2.set_clim(0,Umax)
                        img3.set_clim(0,Umax)
                        img5.set_clim(0,Umax)
                    else:
                        Umax = max(np.amax(Umag2),np.amax(Umag3))
                        img2.set_clim(0,Umax)
                        img3.set_clim(0,Umax)     

                    start_sim(deepcopy(last_yin), deepcopy(last_xo), deepcopy(last_yo), self.geom)
                    img4.set_data(self.geom)
                    quiv4.set_UVC(Ux3[Ys,Xs]*0,-Uy3[Ys,Xs]*0)
                    img4.set_clim(0,1)
                    ax4.set_title('simulation (running)')
                    ax2.set_title(model_A)
                    ax3.set_title(model_B)
                    if third_plot:
                        ax5.set_title(model_C)
                    self.xo = []
                    self.yo = []
                    # print('HERE')

            self.oline.set_data(self.xo, self.yo)
            self.oline.figure.canvas.draw()
        cbar_ax.yaxis.tick_left()

Nplots = 4 + int(third_plot)

plt.ion()
fig, axs = plt.subplots(Nplots, 2)
if third_plot:
    fig.set_size_inches(8, 10)
else:
    fig.set_size_inches(14, 10)

fig.canvas.mpl_connect('key_press_event', on_press)

ax1 = axs[0,0]
img1 = ax1.imshow(geom)
ax1.set_title('geometry')
ax1.set_axis_off()

ax2 = axs[1,0]
img2 = ax2.imshow(geom*0)
quiv2 = ax2.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv2.set_visible(show_vectors)
img2.set_clim(0,1)
ax2.set_title('GAN')
ax2.set_axis_off()

ax3 = axs[2,0]
img3 = ax3.imshow(geom*0)
quiv3 = ax3.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv3.set_visible(show_vectors)
img3.set_clim(0,1)
ax3.set_title('L1')
ax3.set_axis_off()

ax4 = axs[Nplots-1,0]
img4 = ax4.imshow(geom*0)
quiv4 = ax4.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiv4.set_visible(show_vectors)
img4.set_clim(0,1)
ax4.set_title('simulation')
ax4.set_axis_off()

if third_plot:
    ax5 = axs[Nplots-2,0]
    img5 = ax5.imshow(geom*0)
    quiv5 = ax5.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
    quiv5.set_visible(show_vectors)
    img5.set_clim(0,1)
    ax5.set_title('hybrid (1xGAN, 10xL1)')
    ax5.set_axis_off()

fig.subplots_adjust()
poslt = axs[1,0].get_position()
poslb = axs[Nplots-1,0].get_position()
cbar_ax = fig.add_axes([poslb.x0-poslb.width*0.22, poslb.y0, poslb.width*0.1, poslt.y0+poslt.height-poslb.y0])
fig.colorbar(img2, cax=cbar_ax)
cbar_ax.yaxis.tick_left()

oline, = ax1.plot([], [], marker = 'o', color = 'k')  # empty line
inlet, = ax1.plot([], [], marker = 'o', c = 'r', lw=0)  # empty line
live_edit = update_plots(oline,inlet,geom)

axerr1 = axs[0,1]
axerr1.set_axis_off()

axerr2 = axs[1,1]
imgerr2 = axerr2.imshow(geom*0, cmap='Reds')
quiverr2 = axerr2.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiverr2.set_visible(show_vectors)
imgerr2.set_clim(0,1)
axerr2.set_title(model_A+' error')
axerr2.set_axis_off()

axerr3 = axs[2,1]
imgerr3 = axerr3.imshow(geom*0, cmap='Reds')
quiverr3 = axerr3.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
quiverr3.set_visible(show_vectors)
imgerr3.set_clim(0,1)
axerr3.set_title(model_B+' error')
axerr3.set_axis_off()

axerr4 = axs[Nplots-1,1]
axerr4.set_axis_off()

if third_plot:
    axerr5 = axs[Nplots-2,1]
    imgerr5 = axerr5.imshow(geom*0, cmap='Reds')
    quiverr5 = axerr5.quiver(Xs,Ys,Xs*0,Ys*0, scale=20, scale_units='inches', headwidth=4, facecolor="None", lw=0.7, alpha=0.5)
    quiverr5.set_visible(show_vectors)
    imgerr5.set_clim(0,1)
    axerr5.set_title(model_C)
    axerr5.set_axis_off()

# plt.subplots_adjust(wspace=None, hspace=None)

fig.subplots_adjust()
posrt = axs[1,1].get_position()
posrb = axs[Nplots-2,1].get_position()
cbar_ax_err = fig.add_axes([posrb.x0+posrb.width*1.04, posrb.y0, posrb.width*0.1, posrt.y0+posrt.height-posrb.y0])
fig.colorbar(imgerr2, cax=cbar_ax_err)

stop = 0

while True:
    # print('NOW')
    if sim_started == 1 and finished != 2:
        # print('sim_started:')
        # print(sim_started)
        if '100' in os.listdir("simfiles/live_sim/"):
            # print('getLatest')
            get_latest = subprocess.run("./get_latest.sh", cwd='simfiles')
            sim = np.load('simfiles/live_sim/sim_result.npy')
            Usim = sim[:,:,1:3]
            Usim_mag = np.linalg.norm(Usim, axis=2)
            if np.mean(np.abs(Usim_mag-img4.get_array())) < 1e-5:
                stop=stop+1
                if stop == 10:
                    finished = 1
                    print(np.mean(Usim_mag-img4.get_array()))
            img4.set_data(Usim_mag)
            quiv4.set_UVC(sim[Ys,Xs,1],-sim[Ys,Xs,2])
            Umax_sim = max(np.amax(Usim_mag),Umax)
            print(Umax)
            print(Umax_sim)
            img2.set_clim(0,Umax_sim)
            img3.set_clim(0,Umax_sim)
            img4.set_clim(0,Umax_sim)
            if third_plot:
                img5.set_clim(0,Umax_sim)

            Udiff_A = Usim-prediction_A
            Udiff_B = Usim-prediction_B

            L2_A = np.linalg.norm(Udiff_A,axis=2,ord=2)
            L2_B = np.linalg.norm(Udiff_B,axis=2,ord=2)
            
            imgerr2.set_data(L2_A)
            quiverr2.set_UVC(Udiff_A[Ys,Xs,0],-Udiff_A[Ys,Xs,1])
            imgerr3.set_data(L2_B)
            quiverr3.set_UVC(Udiff_B[Ys,Xs,0],-Udiff_B[Ys,Xs,1])

            if third_plot:
                Udiff_C = Usim-prediction_C
                L2_C = np.linalg.norm(Udiff_C,axis=2,ord=2)
                imgerr5.set_data(L2_C)
                quiverr5.set_UVC(Udiff_C[Ys,Xs,0],-Udiff_C[Ys,Xs,1])
                errmax = max(np.amax(L2_A),np.amax(L2_B),np.amax(L2_C))
                imgerr2.set_clim(0,errmax)
                imgerr3.set_clim(0,errmax)
                imgerr5.set_clim(0,errmax)
            else:
                errmax = max(np.amax(L2_A),np.amax(L2_B))
                imgerr2.set_clim(0,errmax)
                imgerr3.set_clim(0,errmax)

            if finished == 1:
                finished = 2
                
                MAE_A = round(np.mean(L2_A),2)
                MAE_B = round(np.mean(L2_B),2)
                if third_plot:
                    MAE_C = round(np.mean(L2_C),2)
                if '5000' in os.listdir("simfiles/live_sim/"):
                    ax4.set_title('simulation (max. # of steps reaches)')
                else:
                    ax4.set_title('simulation (converged)')
                axerr2.set_title('GAN model\nL2 error: %.2f' % MAE_A)
                axerr3.set_title('L1 model\nL2 error: %.2f' % MAE_B)
                # im = Image.fromarray(img1.get_array())
                # im.save("geom.png")
                # im = Image.fromarray(img2.get_array())
                # im.save("prediction.png")
                # im = Image.fromarray(img4.get_array())
                # im.save("sim.png")
                if third_plot:
                    axerr5.set_title('hybrid model\nL2 error: %.2f' % MAE_C)
            elif process.poll() != None:
                finished = 1
        cbar_ax.yaxis.tick_left()

    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    time.sleep(0.1)
