
# import numpy as np
import time

# ------------------------------------------------------------------------------
I = 0b10100
E = 0o15177
_num = pow(2, 14)
_poly = 0b11001
# ------------------------------------------------------------------------------

def dbin_num(n):
    ans = 0
    while n:
        ans += 1
        n //= 2
    return ans

def true_bin(n, l):
    b = bin(n)[2:]
    return ('0' * (l - len(b)) + b)

def xor_div(number, polynom, mat_gen=True):
    to_print = []
    num = bin(number)[2:]
    poly = bin(polynom)[2:]
    div = ''
    res = []
    cur = int(num[:len(poly)], 2)
    to_print.append('_' + num + '|' + poly)
    num = num[len(poly):]
    spaces = ''
    while True:
        # time.sleep(0.5)
        # print(f'{bin(cur)} / {bin(polynom)}')
        if cur // pow(2, len(poly) - 1):
            div += '1'
            cur = cur ^ polynom
            to_print.append(spaces + ' ' + poly)
        else:
            div += '0'
            to_print.append(spaces+ ' ' + (len(bin(res[-1])[2:]) + carry_len + len(_cur))*'0')
        # print('dividing by', div[-1])
        # print('residue', cur, bin(cur))
        res.append(cur)
        if len(num) == 0:
            break
        carry_len = min(len(poly) - dbin_num(cur), len(num))
        if mat_gen:
            carry_len = 1
        cur = int(bin(cur)[2:] + num[:carry_len], 2)
        if mat_gen:
            _cur = '0' * (len(poly) - len(bin(cur)[2:]))
        else:
            _cur = ''
        spaces += ' '*carry_len
        # if carry_len > len(num):
            # res.append(cur)
            # to_print.append(spaces + len(_cur) * ' ' + bin(cur)[2:])
            # break
        to_print.append(spaces + '_' + _cur + bin(cur)[2:])
        num = num[carry_len:]
        # print('remain', num, '\n')
    print(*to_print, sep='\n')
    return res, div

def matmul(I, G):
    res = [0] * len(G[0])
    for x in range(len(G[0])):
        resI = 0
        for y in range(len(G)):
            resI += G[y][x] & I[y]
        res[x] = resI % 2
    return res

print()
_div = xor_div(_num, _poly)
print()
print("RESIDUES")
_residues = []
for i in range(len(_div[0])):
    _res = true_bin(_div[0][i], len(bin(_poly)[2:]) - 1)
    _residues.append(_res)
    print('R{:<2}: {}'.format(i + 1, _res))

print()
matrix = []
for i in range(len(_div[0])):
    matrix.append([0] * len(_div[0]) + list(map(int, _residues[i])))
    matrix[-1][len(_div[0]) - i - 1] = 1
print("MATRIX")
for i in matrix:
    print(*i)

_div_str = true_bin(E, len(matrix[0]))
for i in range(15):
    _div = int(_div_str, 2)
    print()
    print(f'E<-{i}')
    _ans = xor_div(_div, _poly, mat_gen=False)
    W = bin(_ans[0][-1])[2:].count('1')
    if W <= 1:
        print(f'W(R) = {W} <= 1')
        break
    print(f'W(R) = {W} > 1')
    _div_str = _div_str[1:] + _div_str[0]

I = list(map(int, true_bin(I, len(matrix))))
print(I)
print()
print('E = I * G = {}'.format(matmul(I, matrix)))
