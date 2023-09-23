
sudo /opt/ltp/runltp -f containers 2>/dev/null | tee | grep 'Summary:' -A 5 | grep -v 'Summary:' | grep -v -- '--' > log.txt

total=$(cat log.txt | awk '{ sum += $2 } END { print sum }')
passed=$(cat log.txt | grep 'passed' | awk '{ sum += $2 } END { print sum }')

echo 'total:' $total
echo 'passed:' $passed
echo 'ratio:' $(echo $passed/$total | bc -l)
