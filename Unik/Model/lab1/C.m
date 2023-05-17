
A = [
	1 2 3;
	4 5 6;
	7 8 9;
];

B = [
	2 6 8;
	6 11 0.5;
]

C = [13 87 76 45 44];

Mul = A * transpose(B)

D = [A, transpose(B); C]

E = D(1:3, 1:3)

D(2:2:4, :) = [];
D(:, 2:2:5) = []


X = [
	2 5 -8
	5 6 3
	4 -5 -1
];

Y = [8; 12; 23];

Solution = inv(X) * Y
