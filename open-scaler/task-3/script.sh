
for line in $(dpkg-query -L systemd); do test ! -d $line && echo $line; done | grep '^/usr/' | tr -d '\n' | wc -c
