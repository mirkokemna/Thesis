import numpy as np
import sys
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


case = "case{number:04d}"
case = case.format(number=int(sys.argv[1]))

f = open(str(case)+".geo","w")

ffree = 0.05
fwall = 0.02

f.write("SetFactory(\"OpenCASCADE\");\n\n")

## channel
L = 3
D = 1
l = 0.5
d = 1

while not 0.05<d<0.5:
    inlets = np.random.uniform(0,1,2)
    S1 = min(inlets);  # lower shoulder length
    S2 = 1-max(inlets);      # upper shoulder length
    d  = D - S1 - S2;                # small channel width


H  = S1 + d;                     # height above inlet

f.write("ffree = " + str(ffree) + ";\n")
f.write("fwall = " + str(fwall) + ";\n\n")
f.write("L = " + str(L) + ";\n")
f.write("D = " + str(D) + ";\n")
f.write("l = " + str(l) + ";\n")
f.write("S1 = " + str(S1) + ";\n")
f.write("S2 = " + str(S2) + ";\n")
f.write("d = " + str(d) + ";\n")
f.write("H = " + str(H) + ";\n\n")

f.write("// channel\nPoint(1)={l,0,0,ffree};\nPoint(2)={L+l,0,0,ffree};\nPoint(3)={L+l,D,0,ffree};\nPoint(4)={l,D,0,ffree};\nPoint(5)={l,H,0,ffree};\nPoint(6)={0,H,0,ffree};\nPoint(7)={0,S1,0,ffree};\nPoint(8)={l,S1,0,ffree};\n\n")

for i in range(7):
    f.write("Line(%i) = {%i,%i};\n" % (i+1,i+1,i+2))
f.write("Line(8) = {8,1};\n\n")

f.write("Line Loop(100) = {1,2,3,4,5,6,7,8};\n")
f.write("Plane Surface(1) = {100};\n\n")

f.write("//obstacle 1\n")

## obstacle

xs = [0.5]
ys = [0.5]

while np.min(ys) < 0.1 or np.max(ys) > 0.9 or np.min(xs) < 0.6 or np.max(xs) > 3:

    cx = np.random.uniform(1, 2)
    cy = 0
    while abs(cy-D/2) > 0.4:
        cy = np.random.normal(np.mean(inlets),0.2)

    T = 0
    while T < 1 or T > 8:
        T = np.random.normal(2.5,2)



    a = np.random.uniform(30,60)
    b = np.random.uniform(10,100)
    c = np.random.uniform(0,10)
    phi = np.random.uniform(0,2*np.pi)
        
    delta = 0.001

    x = np.arange(-1, 1, delta)
    y = np.arange(-3, 3, delta)
    X, Y = np.meshgrid(x, y)

    Xr = X*np.cos(phi) - Y*np.sin(phi)
    Yr = X*np.sin(phi) + Y*np.cos(phi)

    banana = a*np.power(Xr,2)+b*np.power((Yr+c*np.power(Xr,2)),2)

    CS = plt.contour(X, Y, banana, [T])
    contour = CS.allsegs[0][0]

    contour = contour[:-1,:][np.arange(0,np.shape(contour)[0]-1,10),:]

    xs = contour[:,0] + cx
    ys = contour[:,1] + cy


np.save(case+'_minmax.npy',[np.min(ys),np.max(ys),np.min(xs),np.max(xs)])

np.save(case+'_params.npy',[S1,S2,cx,cy,T,a,b,c,phi])

j = 0
for x in xs:
    f.write("Point(%i) = {%2f,%2f,0,ffree};\n" % (31+j,x,ys[j]))
    j = j+1


Np = len(xs)

f.write("\n")
opoints = list(range(31,31+Np))

for n in range(Np):
    if n == 0:
        f.write("Line(31) = {%i,%i};\n" % (31+Np-1,31))
    else:
        f.write("Line(%i) = {%i,%i};\n" % (31+n,31+n-1,31+n))



f.write("Line Loop(10000) = {")
f.write(','.join(map(str, opoints)))
f.write("};\n\n")

f.write("Plane Surface(30) = {10000};\n")
f.write("BooleanDifference(4) = { Surface{1}; Delete; }{ Surface{30}; Delete; };\n\n")



f.write("extrude_list[] = Extrude {0, 0, 0.1} {Surface{4}; Layers{1}; Recombine;};\n\n")

inlet = 10
outlet = 7
walls = [5,6,8,9,11,12]
obstacle = np.ndarray.tolist(np.asarray(opoints)-18)
front = 13+len(opoints)


f.write("Physical Surface(\"empty\")={4,%i};\n" % (front))
f.write("Physical Surface(\"inlet\")={%i};\n" % (inlet))
f.write("Physical Surface(\"outlet\")={%i};\n" % (outlet))

f.write("Physical Surface(\"walls\")={")
f.write(','.join(map(str, [x for x in walls])))
f.write("};\n")

f.write("Physical Surface(\"obstacle\")={")
f.write(','.join(map(str, [x for x in obstacle])))
f.write("};\n")

f.write("Physical Volume(\"fluidVol\")={1};\n\n")

f.write("Field[1] = Distance;\nField[1].CurvesList = {")
f.write(','.join(map(str, [str(int(x)-3) for x in walls])))
f.write(',')
f.write(','.join(map(str, [str(int(x)-3) for x in obstacle])))
f.write("};\nField[1].Sampling = 100;\n\n")
f.write("Field[2] = Threshold;\nField[2].InField = 1;\nField[2].SizeMin = fwall;\nField[2].SizeMax = ffree;\nField[2].DistMin = 0.01;\nField[2].DistMax = 0.1;\n\n")
f.write("Background Field = 2;\n\n")
f.write("Mesh.MeshSizeExtendFromBoundary = 0;\nMesh.MeshSizeFromPoints = 0;\nMesh.MeshSizeFromCurvature = 0;\n")


f.close()

sys.stdout.write(case) 
