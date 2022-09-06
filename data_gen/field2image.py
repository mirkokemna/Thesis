# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov  3 16:35:05 2021

# @author: mirko
# """

print("create images")

import numpy as np
import codecs
from scipy import interpolate
import matplotlib.image
import os
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry.polygon import Polygon
import sys


case = "case{number:05d}"
case = case.format(number=int(sys.argv[1]))

os.chdir(case)

tend = os.listdir("postProcessing/sampleDict")[0]

with codecs.open('postProcessing/sampleDict/' + tend + '/U_mainsurf.raw', encoding='utf-8-sig') as Ufile:
    Udata = np.loadtxt(Ufile)
with codecs.open('postProcessing/sampleDict/' + tend + '/p_mainsurf.raw', encoding='utf-8-sig') as pfile:
    pdata = np.loadtxt(pfile)

geofile = open(case+".geo")
geolines = geofile.readlines()

L_orig   = 3.5
L_sample = 3
l = 0.5
D = 1

# print(float(geolines[6][5:-2]))

params = np.load(case+'_params.npy')
opoints = np.load(case+'_obstacle.npy')

S1 = params[0]
S2 = params[1]
cx = params[2]
cy = params[3]

d  = D - S1 - S2;                # small channel width
H  = S1 + d;                     # height above inlet

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


# get obstacle coordinates

ob_start = 33

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

if os.path.exists('/mnt/d/UbuntuPortal/obstacles/uniform_gen/arrays/'+case+'.npy'):
  os.remove('/mnt/d/UbuntuPortal/obstacles/uniform_gen/arrays/'+case+'.npy') 
if os.path.exists('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/geom/'+case+'.npy'):
  os.remove('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/geom/'+case+'.npy') 
if os.path.exists('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Ux/'+case+'.npy'):
  os.remove('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Ux/'+case+'.npy') 
if os.path.exists('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Uy/'+case+'.npy'):
  os.remove('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Uy/'+case+'.npy') 
if os.path.exists('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/p/'+case+'.npy'):
  os.remove('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/p/'+case+'.npy') 



tsteps = []
for file in os.listdir():
    if file.isnumeric():
        tsteps.append(int(file))

if np.max(np.array(tsteps)) % 100 == 0 and np.max(np.array(tsteps)) != 2500:
    # print('failed')
    pass

elif np.std(dataset[:,-20:,1]) > 0.01 and np.amax(np.abs(dataset)) < 100: 

    matplotlib.image.imsave('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/geom/'+case+'.png', dataset[:,:,0])
    matplotlib.image.imsave('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Ux/'+case+'.png', dataset[:,:,1])
    matplotlib.image.imsave('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Uy/'+case+'.png', dataset[:,:,2])
    matplotlib.image.imsave('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/Umag/'+case+'.png', np.linalg.norm(dataset[:,:,1:3], axis=2))
    matplotlib.image.imsave('/mnt/d/UbuntuPortal/obstacles/uniform_gen/images/p/'+case+'.png', dataset[:,:,3])

    with open('/mnt/d/UbuntuPortal/obstacles/uniform_gen/arrays/'+case+'.npy', 'wb') as f:
        np.save(f, dataset)

# loaded = np.load('/mnt/d/UbuntuPortal/obstacles/arrays/'+case+'.npy')

os.chdir("..")

# print(np.amax(np.abs(loaded - dataset)))

# plt.imshow(Ux_reg)
# plt.savefig("geom.png")
