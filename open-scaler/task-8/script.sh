
sudo /opt/ltp/runltp -f syscalls -s madvise 2>/dev/null | grep 'Summary:' -A 5 | grep -v 'Summary:' | grep -v -- '--' > log.txt

total=$(cat log.txt | awk '{ sum += $2 } END { print sum }')
passed=$(cat log.txt | grep 'passed' | awk '{ sum += $2 } END { print sum }')

echo 'total:' $total
echo 'passed:' $passed
echo 'ratio:' $(echo $passed/$total | bc -l)
