base="/workspace"
cd $base/sources
make template
echo > $base/result
for idx in {0..9}
do
    echo "Testcase $idx"
    cd $base/sources/ListScan/Dataset/$idx
    ./../../../ListScan_template -e output.raw -i input.raw -o o.raw -t vector >> $base/result
    echo >> $base/result
done