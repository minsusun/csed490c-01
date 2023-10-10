base="/workspace/Lab1_cuda/sources"
cd $base
make template
make dataset_generator
echo > $base/result
for idx in {0..9}
do
    cd $base/VectorAdd/Dataset/$idx
    ./../../../VectorAdd_template -e output.raw -i input0.raw,input1.raw -o o.raw -t vector >> $base/result
    echo >> $base/result
done