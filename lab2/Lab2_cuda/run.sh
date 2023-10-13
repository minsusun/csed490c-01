base="/workspace/lab2/Lab2_cuda/sources"
for TILE_WIDTH in {2,4,8,12,16,24,32}
do
    cd $base
    sed -i "3c\#define TILE_WIDTH $TILE_WIDTH" template.cu
    make template
    echo > $base/result_$TILE_WIDTH
    for idx in {0..8}
    do
        cd $base/TiledMatrixMultiplication/Dataset/$idx
        ./../../../TiledGEMM_template -e output.raw -i input0.raw,input1.raw -o o.raw -t matrix >> $base/result_$TILE_WIDTH
        echo >> $base/result
    done
done