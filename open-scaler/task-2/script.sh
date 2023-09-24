
yumdownloader -v --source --nogpgcheck --refresh --downloadonly --resolve sssd 2> /dev/null > /dev/null

mock --rebuild sssd-2.6.1-9.os2203sp2.src.rpm --resultdir=result

