
figure
clf
hold on
grid on

t = linspace(-5, 5, 100);
plot(sin(t), 2.*cos(2.*t), 'LineWidth', 2);


figure
clf
hold on
grid on

x = linspace(0, 3, 20);
[X,Y] = meshgrid(x, x);
Z = (X.^2 + Y.^2) ./ (5 .* X .* Y) + log(X + Y);
surf(X, Y, Z);
view(60, 60);

