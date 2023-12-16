cd ../
git submodule update --init
cd Project/pbbsbench
git submodule update --init
cd testData/geometryData/data
make 2DinCube_1000000
make 2Dkuzmin_1000000
make 3DonSphere_1000000
make 3DinCube_1000000
make 3Dplummer_1000000
cd ../../../../
cp -r pbbsbench/testData/geometryData/data ./
rm data/Makefile
mkdir bin data output