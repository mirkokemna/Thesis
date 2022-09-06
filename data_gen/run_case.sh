#!/bin/bash
[ -d $1 ] && rm -r $1
cp -r base_case $1
mv $1* $1
cd $1
gmsh $1.geo -3 -o $1.msh -format msh2 > gmshlog
gmshToFoam $1.msh > gmsh2foamlog
perl -0777 -i -pe 's/empty\n    {\n        type            patch;/empty\n    {\n        type            empty;/igs' constant/polyMesh/boundary
perl -0777 -i -pe 's/walls\n    {\n        type            patch;/walls\n    {\n        type            wall;/igs' constant/polyMesh/boundary
perl -0777 -i -pe 's/obstacle\n    {\n        type            patch;/obstacle\n    {\n        type            wall;/igs' constant/polyMesh/boundary
simpleFoam > solvelog
simpleFoam -postProcess -func yPlus -latestTime > yPluslog
postProcess -func sampleDict -latestTime
