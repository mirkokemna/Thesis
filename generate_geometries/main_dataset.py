import numpy as np
from shapely.geometry import Point, Polygon

def generate_case(casenum):
    case = "case{number:05d}"
    case = case.format(number=int(casenum))

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

    f.write("//obstacle\n")

    admissable = False

    while not admissable:
        ## obstacle
        cx = 0
        while L+l-1 < cx or cx < l+0.5:
            cx = np.random.uniform(1, 2.5)
        cy = 0
        while abs(cy-D/2) > 0.4:
            cy = np.random.normal(D/2,0.4)

        Np = np.random.randint(3,10)

        phis = np.sort(np.random.uniform(0, 2*np.pi, size=Np))
        
        rs   = np.random.uniform(0.1, 0.6, size=Np)

        scaling = np.random.uniform(0.5, 1.5)
        coin = np.random.randint(0,2)

        if coin:
            scaling = 1/scaling

        xs = cx + rs*np.cos(phis)*scaling
        ys = cy + rs*np.sin(phis)*scaling


        tops = np.where(ys>0.95)[0]
        bottoms = np.where(ys<0.05)[0]

        ys[tops] = 1
        ys[bottoms] = 0
        
        if np.size(tops)>1:
            jump = (tops[-1]-tops[0])%(Np-1)
            xs = np.delete(xs,np.arange(tops[0]+1,(tops[0]+jump)%(Np-1)))
            ys = np.delete(ys,np.arange(tops[0]+1,(tops[0]+jump)%(Np-1)))
        if np.size(bottoms)>1:
            jump = (bottoms[-1]-bottoms[0])%(Np-1)
            xs = np.delete(xs,np.arange(bottoms[0]+1,(bottoms[0]+jump)%(Np-1)))
            ys = np.delete(ys,np.arange(bottoms[0]+1,(bottoms[0]+jump)%(Np-1)))


        if 1-(max(ys)-min(ys)) < d:
            continue

        if np.logical_or(np.any(xs<0.6), np.any(xs>3)):
            continue

        Np = np.size(xs)


        phis = np.arctan2(ys-cy,xs-cx)
        sorting = np.argsort(phis)

        xs = xs[sorting]
        ys = ys[sorting]

        cp = Point(cx, cy)

        coords = np.transpose(np.vstack((xs,ys)))
        obstacle = Polygon(coords)

        if not obstacle.contains(cp):
            continue

        mindeltaphi = np.min(np.abs(phis-np.append(phis[1:],phis[0]))%np.pi)/np.pi*180

        if  mindeltaphi< 10:
            continue

        admissable = True
        

    if np.size(bottoms)>0:
        wallcase = 2
        if np.size(bottoms)>1:
            wallcase = 4

    elif np.size(tops)>0:
        wallcase = 3
        if np.size(tops)>1:
            wallcase = 5

    else:
        wallcase = 1


    j = 0
    for x in xs:
        f.write("Point(%i) = {%2f,%2f,0,ffree};\n" % (11+j,x,ys[j]))
        j = j+1


    f.write("\n")


    opoints = list(range(11,11+Np))

    for n in range(Np):
        if n == 0:
            f.write("Line(11) = {%i,%i};\n" % (11+Np-1,11))
        else:
            f.write("Line(%i) = {%i,%i};\n" % (11+n,11+n-1,11+n))


    f.write("Line Loop(100) = {1,2,3,4,5,6,7,8};\n")
    f.write("Line Loop(200) = {")
    f.write(','.join(map(str, opoints)))
    f.write("};\n\n")

    f.write("Plane Surface(1) = {100};\n")
    f.write("Plane Surface(2) = {200};\n")
    f.write("BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };\n\n")

    f.write("extrude_list[] = Extrude {0, 0, 0.1} {Surface{3}; Layers{1}; Recombine;};\n\n")

    if wallcase == 1:
        inlet = 9
        outlet = 6
        walls = [4,5,7,8,10,11]
        obstacle = np.ndarray.tolist(np.asarray(opoints)+1)
        front = 12+Np
    elif wallcase == 2:
        front = 13+Np
        inlet = 6
        outlet = 10
        walls = [4,5,7,8,9,11,12]
        obstacle = np.ndarray.tolist(np.asarray(opoints)+2)
    elif wallcase == 3:
        front = 13+Np
        inlet = 6
        outlet = 11
        walls = [4,5,7,8,9,10,12]
        obstacle = np.ndarray.tolist(np.asarray(opoints)+2)
    elif wallcase == 4:
        front = 12+Np
        inlet = 6
        outlet = 10
        walls = [4,5,7,8,9,11,11+Np]
        obstacle = opoints[1:]
    elif wallcase == 5:
        front = 12+Np
        inlet = 6
        outlet = 10+Np
        walls = [4,5,7,8,9,9+Np,11+Np]
        obstacle = np.ndarray.tolist(np.asarray(opoints[:-1])-1)



    np.save(case + '_params.npy', [S1,S2,cx,cy])
    np.save(case + '_obstacle.npy', coords)

    f.write("// wallcase: %i\n" % (wallcase))

    f.write("// center coordinates = (%.2f,%.2f)" % (cx,cy))

    f.write("// x-coordinates = [")
    f.write(','.join(map(str, [x for x in xs])))
    f.write("]\n")

    f.write("// y-coordinates = [")
    f.write(','.join(map(str, [y for y in ys])))
    f.write("]\n")


    f.write("Physical Surface(\"empty\")={3,%i};\n" % (front))
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

    return case
