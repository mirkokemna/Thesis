# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov  3 16:35:05 2021

# @author: mirko
# """

print("create images")

import numpy as np
import codecs
from scipy import interpolate
import os
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry.polygon import Polygon
from os.path import exists

case = "live_sim"

os.chdir(case)



steps = os.listdir("postProcessing/sampleDict")
tend = np.max(np.asarray(steps, dtype=int))

print('______________________________________________')
print(tend)
print('______________________________________________')


with codecs.open('postProcessing/sampleDict/' + str(tend) + '/U_mainsurf.raw', encoding='utf-8-sig') as Ufile:
    Udata = np.loadtxt(Ufile)
with codecs.open('postProcessing/sampleDict/' + str(tend) + '/p_mainsurf.raw', encoding='utf-8-sig') as pfile:
    pdata = np.loadtxt(pfile)

geofile = open(case+".geo")
geolines = geofile.readlines()

L_orig   = 3.5
L_sample = 3
l = 0.5
D = 1

# print(float(geolines[6][5:-2]))

S1 = float(geolines[9][5:-2])
S2 = float(geolines[10][5:-2])
d = float(geolines[11][4:-2])
H = float(geolines[12][4:-2])


X  = Udata[:,0]
Y  = Udata[:,1]
XY = Udata[:,0:2]

Ux = Udata[:,3]
Uy = Udata[:,4]
p  = pdata[:,3]

f = 256/3 # samples per meter

Nx = int(f*L_sample)
Ny = int(f*D)

x = np.linspace(0.01,L_sample-0.01,Nx)
y = np.linspace(0.01,D-0.01,Ny)
xx,yy = np.meshgrid(x,y)

dataset = np.ones(np.append(np.shape(xx),4))
nonans = np.ones(np.append(np.shape(xx),4))

dataset[:,:,0] = np.ones(np.shape(xx))
dataset[:,:,1] = interpolate.griddata(XY, Ux, (xx,yy),method='cubic')
dataset[:,:,2] = interpolate.griddata(XY, Uy, (xx,yy),method='cubic')
dataset[:,:,3] = interpolate.griddata(XY, p, (xx,yy),method='cubic')
nonans[:,:,1] = interpolate.griddata(XY, Ux, (xx,yy),method='nearest')
nonans[:,:,2] = interpolate.griddata(XY, Uy, (xx,yy),method='nearest')
nonans[:,:,3] = interpolate.griddata(XY, p, (xx,yy),method='nearest')
dataset[np.where(np.isnan(dataset))] = nonans[np.where(np.isnan(dataset))]
# dataset[:,:,3] = dataset[:,:,3] - np.amin(nonans[:,:,3])

# Ux_regn = interpolate.griddata(XY, Ux, (xx,yy),method='nearest')
# Uy_regn = interpolate.griddata(XY, Uy, (xx,yy),method='nearest')
# p_regn  = interpolate.griddata(XY, p, (xx,yy),method='nearest')



# reset points outside walls
dataset[np.logical_and(xx < l, yy < S1),:] = 0
dataset[np.logical_and(xx < l, yy > H), :] = 0

Nline = geolines[2]
Nvertices = ''
for c in Nline:
    if c.isdigit():
        Nvertices = Nvertices + c

if int(Nvertices) > 2:
    # get obstacle coordinates
    opoints = np.empty((0,2))

    ob_start = 34

    i = 0
    while geolines[ob_start+i][0] == 'P':
        j = 0
        while geolines[ob_start+i][j] != '{':
            j = j+1
        j = j+1
        x1 = j
        while geolines[ob_start+i][j] != ',':
            j = j+1
        x2 = j
        j = j+1
        y1 = j
        while geolines[ob_start+i][j] != ',':
            j = j+1
        y2 = j
        xp = float(geolines[ob_start+i][x1:x2])
        yp = float(geolines[ob_start+i][y1:y2])
        opoints = np.vstack((opoints,[xp,yp]))
        i = i+1


    xxre = np.reshape(xx, np.size(xx))
    yyre = np.reshape(yy, np.size(yy))

    xxyy = np.transpose(np.vstack((xxre,yyre)))
    obs_logic = np.ones(np.size(xx))

    polygon = Polygon(opoints)
    for i in range(np.size(xx)):
        obs_logic[i] = polygon.contains(Point(xxyy[i,:]))

    obs_logic = np.reshape(obs_logic, np.shape(xx)) == 1


    dataset[obs_logic,:] = 0

# dataset = np.empty[]
# plt.imshow(dataset[:,:,3])
# plt.colorbar()
# plt.savefig("test.png")

# print(dataset[:,:,1])
# dataset[:,:,0] = dataset[:,:,1]



# if tend % 100 == 0 and tend != 2500:
#     # print('failed')
#     pass

if np.std(dataset[:,-20:,1]) > 0.01 and np.amax(np.abs(dataset)) < 100: 

    with open('sim_result.npy', 'wb') as f:
        print(os.getcwd())
        np.save(f, dataset)

# loaded = np.load('/mnt/d/UbuntuPortal/obstacles/arrays/'+case+'.npy')


# print(np.amax(np.abs(loaded - dataset)))

# plt.imshow(Ux_reg)
# plt.savefig("geom.png")
