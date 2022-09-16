

function f = factorial(n)
	if n < 0
		disp('you suck!');
	elseif n == 0
		f = 1;
		return ;
	else
		f = n * factorial(n - 1);
		return ;
	end
end

