
for line in $(rpm -ql systemd); do test ! -d $line && echo $line; done | grep '^/usr/' | tr -d '\n' | wc -c
