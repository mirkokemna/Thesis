import numpy as np
import sys
from shapely.geometry.polygon import Polygon


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
cx = 0
while L+l-1 < cx or cx < l+0.5:
    cx = np.random.uniform(1, 2)
cy = 0
while abs(cy-D/2) > 0.4:
    cy = np.random.normal(np.mean(inlets),0.2)

phi0 = np.random.uniform(0, 2*np.pi)
phis = []
xs   = []
ys   = []
phis.append(phi0)
i = 0
dg2rad = 2*np.pi/360
pphi = phi0

reset = []
side = -1

out_bounds = []

while True:
    px = 0
    pr = -1
    while px<0.6 or px>3:
        pr = np.random.uniform(.1,.5)
        pphi =  np.random.uniform(phis[-1]+30*dg2rad, phis[-1]+150*dg2rad)
        px = cx + pr*np.cos(pphi)
    py = cy + pr*np.sin(pphi)

    if pphi >= phis[min([1,i])]+2*np.pi:
        # print('BROKEN')
        if len(xs)>2:
            break
        else:
            phis = []
            phis.append(phi0)
            xs = []
            ys = []
            i = 0
            dg2rad = 2*np.pi/360
            pphi = 0
            reset = []
            side = -1
            out_bounds = []
            continue

    if py <= 0.05:
        py = 0.05
        side = 0
        out_bounds.append(i)
        if len(reset) > 1:
            ys = ys[:reset[1]]
            xs = xs[:reset[1]]
            phis = phis[:reset[1]+1]
            i = reset[1]
            reset = reset[:reset[1]]

    elif py >= D-0.05:
        py = D-0.05
        side = 1
        out_bounds.append(i)
        if len(reset) > 1:
            ys = ys[:reset[1]]
            xs = xs[:reset[1]]
            phis = phis[:reset[1]+1]
            i = reset[1]
            reset = reset[:reset[1]]
            # print('reset')

    if side != -1:
        reset.append(i)

    phis.append(pphi)
    xs.append(px)
    ys.append(py)

    if max(ys)-min(ys) > 0.95:
        # print('BIG RESET')
        phis = []
        phis.append(phi0)
        xs = []
        ys = []
        i = 0
        dg2rad = 2*np.pi/360
        pphi = 0
        reset = []
        side = -1
        out_bounds = []
        continue

    i = i+1
    

j = 0
for x in xs:
    f.write("Point(%i) = {%2f,%2f,0,ffree};\n" % (11+j,x,ys[j]))
    j = j+1


Np = len(xs)

f.write("\n")

opoints_1 = list(range(11,11+Np))

for n in range(Np):
    if n == 0:
        f.write("Line(11) = {%i,%i};\n" % (11+Np-1,11))
    else:
        f.write("Line(%i) = {%i,%i};\n" % (11+n,11+n-1,11+n))


f.write("Line Loop(200) = {")
f.write(','.join(map(str, opoints_1)))
f.write("};\n\n")

f.write("Plane Surface(2) = {200};\n")
f.write("BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };\n\n")

coords_1 = np.empty((Np,2))
coords_1[:,0] = xs
coords_1[:,1] = ys
polygon_1 = Polygon(coords_1)









f.write("//obstacle 2\n")
## obstacle
cx = 0
while L+l-1 < cx or cx < l+0.5:
    cx = np.random.uniform(1, 2)
cy = 0
while abs(cy-D/2) > 0.4:
    cy = np.random.normal(np.mean(inlets),0.2)

phi0 = np.random.uniform(0, 2*np.pi)
phis = []
xs   = []
ys   = []
phis.append(phi0)
i = 0
dg2rad = 2*np.pi/360
pphi = phi0

reset = []
side = -1

out_bounds = []

while True:
    px = 0
    pr = -1
    while px<0.6 or px>3:
        pr = np.random.uniform(.1,.5)
        pphi =  np.random.uniform(phis[-1]+30*dg2rad, phis[-1]+150*dg2rad)
        px = cx + pr*np.cos(pphi)
    py = cy + pr*np.sin(pphi)
    if pphi >= phis[min([1,i])]+2*np.pi:
        if len(xs)>2:
            coords_2 = np.empty((np.size(xs),2))
            coords_2[:,0] = xs
            coords_2[:,1] = ys
            polygon_2 = Polygon(coords_2)
            distance = polygon_1.distance(polygon_2)
            if distance > 0.05:
                break

        phis = []
        phis.append(phi0)
        xs = []
        ys = []
        i = 0
        dg2rad = 2*np.pi/360
        pphi = 0
        reset = []
        side = -1
        out_bounds = []
        cx = 0
        while L+l-1 < cx or cx < l+0.5:
            cx = np.random.uniform(1, 2)
        cy = 0
        while abs(cy-D/2) > 0.4:
            cy = np.random.normal(np.mean(inlets),0.2)
        continue

    if py <= 0.05:
        py = 0.05
        side = 0
        out_bounds.append(i)
        if len(reset) > 1:
            ys = ys[:reset[1]]
            xs = xs[:reset[1]]
            phis = phis[:reset[1]+1]
            i = reset[1]
            reset = reset[:reset[1]]
            # print('reset')

    elif py >= D-0.05:
        py = D-0.05
        side = 1
        out_bounds.append(i)
        if len(reset) > 1:
            ys = ys[:reset[1]]
            xs = xs[:reset[1]]
            phis = phis[:reset[1]+1]
            i = reset[1]
            reset = reset[:reset[1]]

    if side != -1:
        reset.append(i)

    phis.append(pphi)
    xs.append(px)
    ys.append(py)

    if max(ys)-min(ys) > 0.95:
        phis = []
        phis.append(phi0)
        xs = []
        ys = []
        i = 0
        dg2rad = 2*np.pi/360
        pphi = 0
        reset = []
        side = -1
        out_bounds = []
        continue

    i = i+1
    

j = 0
for x in xs:
    f.write("Point(%i) = {%2f,%2f,0,ffree};\n" % (31+j,x,ys[j]))
    j = j+1


Np = len(xs)

f.write("\n")


opoints_2 = list(range(31,31+Np))

for n in range(Np):
    if n == 0:
        f.write("Line(31) = {%i,%i};\n" % (31+Np-1,31))
    else:
        f.write("Line(%i) = {%i,%i};\n" % (31+n,31+n-1,31+n))


f.write("Line Loop(300) = {")
f.write(','.join(map(str, opoints_2)))
f.write("};\n\n")

f.write("Plane Surface(30) = {300};\n")
f.write("BooleanDifference(4) = { Surface{3}; Delete; }{ Surface{30}; Delete; };\n\n")




opoints = opoints_1
for i in range(len(opoints_2)):
    opoints.append(int(opoints[-1])+1)


f.write("extrude_list[] = Extrude {0, 0, 0.1} {Surface{4}; Layers{1}; Recombine;};\n\n")

inlet = 10
outlet = 7
walls = [5,6,8,9,11,12]
obstacle = np.ndarray.tolist(np.asarray(opoints)+2)
front = 13+len(opoints)
wallcase = 1


f.write("// wallcase: %i\n" % (wallcase))
f.write("// out_bounds = [")
f.write(','.join(map(str, [x for x in out_bounds])))
f.write("]\n")

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

f.write("Point(100) = {%2f,%2f,0,ffree};\n" % (cx,cy))

f.close()

sys.stdout.write(case) 
