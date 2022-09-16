
x = linspace(0, 10, 100);

figure
clf
hold on
grid on
fplot(@(x)(x - sin(3.*x)), 'LineWidth', 2);

figure
hold on
grid on
plot(x, x ./ sin(x), 'LineWidth', 2);
legend('x - sin(3x)');


phi = linspace(0, pi, 300);
phi2 = linspace(-2*pi, 2*pi, 300);
f = @(phi)(phi .* sin(5.*phi));

figure
clf
hold on
grid on
polar(phi, f(phi));

figure
hold on
grid on
polar(phi2, f(phi2));


figure
clf
hold on
grid on

xlabel('Pyton');
ylabel('IsBetter');
a = 5;
b = 4;
x = linspace(0, 2*pi, 100);
plot(a.*cos(x) - 3, b.*sin(x) + 2, 'LineWidth', 2, 'Color', [0,1,0]);
plot([-10, 10], [0, 0], 'k');
plot([0, 0], [-10, 10], 'k');

