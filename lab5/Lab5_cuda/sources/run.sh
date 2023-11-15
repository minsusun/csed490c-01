base="/workspace"
cd $base/sources
make template
echo > $base/result
for idx in {0..7}
do
    echo "Testcase $idx"
    cd $base/sources/Histogram/Dataset/$idx
    ./../../../Histogram_template -e output.raw -i input.raw -o o.raw -t integral_vector >> $base/result
    echo >> $base/result
done