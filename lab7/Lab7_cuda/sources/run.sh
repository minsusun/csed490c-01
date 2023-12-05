base="/workspace/Lab7_cuda"
cd $base/sources
make template
echo > $base/result
for idx in {0..8}
do
    echo "Testcase $idx"
    cd $base/sources/SparseMV/Dataset/$idx
    ./../../../JDS_T_template -e output.raw -i input0.raw,input1.raw -o o.raw -t vector >> $base/result
    echo >> $base/result
done