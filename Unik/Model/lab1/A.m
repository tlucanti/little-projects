

f = @(x, y)((log(sin(x * pi / 8)) + x*x + sqrt(2*x - 2))) / (2 ^ y);

x = 1.001;
y = -10.7;

F1 = f(x, y)

x = 2 + 3j;
y = 1 - j;

F2 = f(x, y)


