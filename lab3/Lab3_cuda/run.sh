base="/workspace/csed490c-01/Lab3_cuda"
for streamSize in {1..32}
do
    cd $base/sources
    sed -i "3c\#define N_STREAM $streamSize" template.cu
    make template
    echo > $base/result_$streamSize
    for idx in {0..9}
    do
        cd $base/sources/VectorAdd/Dataset/$idx
        ./../../../StreamVectorAdd_template -e output.raw -i input0.raw,input1.raw -o o.raw -t vector >> $base/result_$streamSize
        echo >> $base/result_$streamSize
    done
done