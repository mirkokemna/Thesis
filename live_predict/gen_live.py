import numpy as np
import os

def create_geo(yin, xs, ys):

    # print(os.getcwd())
    f = open("simfiles/live_sim.geo","w")

    ffree = 0.05
    fwall = 0.02

    f.write("SetFactory(\"OpenCASCADE\");\n\n")
    f.write("//Vertices:%i\n" % (len(xs)))

    ## channel
    L = 3.5;
    D = 1;
    l = 0.5;

    S1 = min(yin);  # lower shoulder length
    S2 = 1-max(yin);      # upper shoulder length
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

    # print(len(xs))

    side = -1
    out_bounds = []

    if len(xs)>2:

        f.write("//obstacle\n")
        ## obstacle

        j = 0
        for x in xs:
            f.write("Point(%i) = {%2f,%2f,0,ffree};\n" % (11+j,x,ys[j]))
            j = j+1

        Np = len(xs)
        # print(reset)

        f.write("\n")

        # print(side)
        # print("Np="+str(Np))
        # print(out_bounds)

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

        wallcase = -1

        if side == -1:
            inlet = 9
            outlet = 6
            walls = [4,5,7,8,10,11]
            obstacle = np.ndarray.tolist(np.asarray(opoints)+1)
            front = [3,12+Np]
            wallcase = 1
        elif len(out_bounds) == 1:
            front = [3,13+Np]
            if side == 0:
                inlet = 6
                outlet = 10
                walls = [4,5,7,8,9,11,12]
                obstacle = np.ndarray.tolist(np.asarray(opoints)+2)
                wallcase = 2
            elif side == 1:
                inlet = 6
                outlet = 11
                walls = [4,5,7,8,9,10,12]
                obstacle = np.ndarray.tolist(np.asarray(opoints)+2)
                wallcase = 3
        elif side == 0:
            front = [3,12+Np]
            inlet = 6
            outlet = 10
            walls = [4,5,7,8,9,11,11+Np]
            obstacle = opoints[1:]
            wallcase = 4
        elif side == 1:
            front = [3,12+Np]
            inlet = 6
            outlet = 10+Np
            walls = [4,5,7,8,9,9+Np,11+Np]
            obstacle = np.ndarray.tolist(np.asarray(opoints[:-1])-1)
            wallcase = 5

    else:
        f.write("Line Loop(100) = {1,2,3,4,5,6,7,8};\n")
        f.write("Plane Surface(1) = {100};\n")
        f.write("extrude_list[] = Extrude {0, 0, 0.1} {Surface{1}; Layers{1}; Recombine;};\n\n")
        inlet = 7
        outlet = 3
        walls = [2,4,5,6,8,9]
        front = [1,10]
        wallcase = 1
        obstacle = []

    # print(wallcase)
    # print(out_bounds)

    f.write("// wallcase: %i\n" % (wallcase))
    f.write("// out_bounds = [")
    f.write(','.join(map(str, [x for x in out_bounds])))
    f.write("]\n")

    f.write("Physical Surface(\"empty\")={%i,%i};\n" % (front[0],front[1]))
    f.write("Physical Surface(\"inlet\")={%i};\n" % (inlet))
    f.write("Physical Surface(\"outlet\")={%i};\n" % (outlet))

    f.write("Physical Surface(\"walls\")={")
    f.write(','.join(map(str, [x for x in walls])))
    f.write("};\n")

    f.write("Physical Surface(\"obstacle\")={")
    f.write(','.join(map(str, [x for x in obstacle])))
    f.write("};\n")

    f.write("Physical Volume(\"fluidVol\")={1};\n\n")

    if len(obstacle) > 0:
        f.write("Field[1] = Distance;\nField[1].CurvesList = {")
        f.write(','.join(map(str, [x for x in walls])))
        f.write(',')
        f.write(','.join(map(str, [str(int(x)-3) for x in obstacle])))
    else:
        f.write("Field[1] = Distance;\nField[1].CurvesList = {1,3,4,5,7,8")
    f.write("};\nField[1].Sampling = 100;\n\n")
    f.write("Field[2] = Threshold;\nField[2].InField = 1;\nField[2].SizeMin = fwall;\nField[2].SizeMax = ffree;\nField[2].DistMin = 0.01;\nField[2].DistMax = 0.1;\n\n")
    f.write("Background Field = 2;\n\n")
    f.write("Mesh.MeshSizeExtendFromBoundary = 0;\nMesh.MeshSizeFromPoints = 0;\nMesh.MeshSizeFromCurvature = 0;\n")

    f.close()