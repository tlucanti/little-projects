#!/bin/zsh

echo "===== test simple expressions ====="
./lab4 <<< 'x = 0;'
./lab4 <<< 'x = 1 + 1;'
./lab4 <<< 'x = 1 + 2 * 3 + 4;';
./lab4 <<< 'x = 1 / 2 * 3 - 4 + 5;';

echo "===== test multiple expressions ====="
./lab4 <<< 'x = 1 + 1; y = x;'
./lab4 <<< 'x = 1; y = x; z = y; w = x + y + z;'

echo "===== test brackets ====="
./lab4 <<< 'x = (1);'
./lab4 <<< 'x = (1 + 1);'
./lab4 <<< 'x = 1 + (1) - 4;'
./lab4 <<< 'x = (0); y = (x);'
./lab4 <<< 'x = (1 + 1) * (2 + 3);'
./lab4 <<< 'x = (1 * 2) / (3 + 4) + (4 * 4) + 5;'
./lab4 <<< 'x = (((((1)))));'
./lab4 <<< 'x = ((1)) + ((2));'
./lab4 <<< 'x = 1 + (1 + (1 + (1 + 1)));'
./lab4 <<< 'x = ((((1 + 1) + 1) + 1) + 1) + 1;'

echo "===== syntax error test ====="
./lab4 <<< ''
./lab4 <<< ';'
./lab4 <<< 'lol'
./lab4 <<< 'kek = kek;'
./lab4 <<< '123 = 123;'
./lab4 <<< 'x = ;'
./lab4 <<< 'x = );'
./lab4 <<< 'x = (1;'
./lab4 <<< 'x = 1x;'
./lab4 <<< 'x = 1 + + 4;'
./lab4 <<< 'x = 1 1 1;'
./lab4 <<< '+ +;'
./lab4 <<< 'x x = y y;'
./lab4 <<< 'x = y y;'

