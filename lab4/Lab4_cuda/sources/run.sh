base="/workspace/csed490c-01/Lab4_cuda"
cd $base/sources
make template
echo > $base/result
for idx in {0..6}
do
    cd $base/sources/Convolution/Dataset/$idx
    ./../../../Convolution_template -e output.ppm -i input0.ppm,input1.raw -o o.ppm -t image >> $base/result
    echo >> $base/result
done