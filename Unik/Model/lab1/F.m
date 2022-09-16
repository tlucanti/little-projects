
fac(10)

s = 0;
N = 2;
acc = 0.001;
i = 1

while 1
	s_new = s + 1/(i^N);
	if abs(s - s_new) < acc
	    s = s_new;
		break ;
	end
	s = s_new;
	i = i + 1;
end

s

