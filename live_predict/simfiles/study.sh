#!/bin/bash

echo "starting simulation"

# echo $i
cp -r base_case live_sim
mv live_sim.geo live_sim
cd live_sim
gmsh live_sim.geo -3 -o live_sim.msh -format msh2 > gmshlog
gmshToFoam live_sim.msh > gmsh2foamlog
perl -0777 -i -pe 's/empty\n    {\n        type            patch;/empty\n    {\n        type            empty;/igs' constant/polyMesh/boundary
perl -0777 -i -pe 's/walls\n    {\n        type            patch;/walls\n    {\n        type            wall;/igs' constant/polyMesh/boundary
perl -0777 -i -pe 's/obstacle\n    {\n        type            patch;/obstacle\n    {\n        type            wall;/igs' constant/polyMesh/boundary
simpleFoam > solvelog
touch "done.btch"
cd ..
echo "simulation completed"