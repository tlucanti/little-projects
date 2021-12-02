
# ------------------------------------------------------------------------------
POLY_SIZE   = 15        # polynom size: x^15
POLYNOM     = 0b10011   # binary polynom: x5-x4-x3-x2-x1-x0
E           = 0o15234   # octal input
# ------------------------------------------------------------------------------
T           = 1

def true_bin(n, l):
    b = bin(n)[2:]
    return '0' * (l - len(b)) + b


def bin_len(n):
    ans = 0
    while n:
        n //= 2
        ans += 1
    return ans


def mat_gen(poly_size, polynom):
    partials = []
    residues = []
    poly_str = bin(polynom)[2:]

    poly_max = pow(2, len(poly_str) - 1)
    res = poly_max
    for i in range(poly_size - len(poly_str) + 1):
        if res // poly_max:
            partials.append(1)
            res ^= polynom
        else:
            partials.append(0)
        residues.append(res)
        res <<= 1
    return residues, partials


def xor_div(e, polynom):
    residues = []
    shift = []
    poly_str = bin(polynom)[2:]
    poly_len = len(poly_str)
    e_str = bin(e)[2:]

    poly_max = pow(2, len(poly_str) - 1)
    res = int(e_str[:poly_len], 2)
    e_str = e_str[poly_len:]
    while True:
        if not res // poly_max:
            break
        res ^= polynom
        carry = min(poly_len - bin_len(res), len(e_str))
        shift.append(carry)
        if res == 0:
            while len(e_str) and e_str[0] == '0':
                e_str = e_str[1:]
                shift[-1] += 1
        for c in range(carry):
            if len(e_str) == 0:
                break
            res <<= 1
            if e_str[0] == '1':
                res += 1
            e_str = e_str[1:]
        residues.append(res)
    return residues, shift


def matmul(I, G):
    res = [0] * len(G[0])
    for x in range(len(G[0])):
        resI = 0
        for y in range(len(G)):
            resI += G[y][x] & I[y]
        res[x] = resI % 2
    return res

print()
print('-------------------------------------------------------')
print('Finding an error with inputs:')
print(f'Polynom = {bin(POLYNOM)[2:]}')
print(f'E       = {bin(E)[2:]}(2) = {oct(E)[2:]}(8)')
print('-------------------------------------------------------')

print()
print('DIVISION')
poly_size = len(bin(POLYNOM)) - 2
res, div = mat_gen(POLY_SIZE, POLYNOM)
matr_size = len(res)
print(f'_{pow(10, POLY_SIZE)}|{bin(POLYNOM)[2:]}')
for i in range(matr_size - 1):
    print(' ' * (i+1), bin(POLYNOM)[2:] if div[i] else '0' * poly_size, sep='')
    print(' ' * (i+1), '_', true_bin(res[i], poly_size - 1), '0', sep='')
print(' ' * matr_size, bin(POLYNOM)[2:] if div[-1] else '0' * poly_size, sep='')
print(' ' * (matr_size + 1), true_bin(res[-1], poly_size - 2), sep='')

print()
print('RESIDUES')
for i in range(matr_size):
    print('R{:<2}: {}'.format(i + 1, true_bin(res[i], poly_size - 1)))

print()
print('MATRIX')
matrix = []

for i in range(matr_size):
    matrix.append([0] * matr_size + list(true_bin(res[i], poly_size - 1)))
    matrix[-1] = list(map(int, matrix[-1]))
    matrix[-1][matr_size - i - 1] = 1
    print(*matrix[-1])

E_shift = E
message_size = len(matrix[0])
ok = False
for i in range(15):
    print()
    print(f'E <- {i}')
    res, sh = xor_div(E_shift, POLYNOM)

    space = 0
    print(f'_{bin(E_shift)[2:]}|{bin(POLYNOM)[2:]}')
    for j in range(len(res) - 1):
        print(' ' * space, bin(POLYNOM)[2:])
        space += sh[j]
        print(' ' * space, '_', bin(res[j])[2:], sep='')
    print(' ' * space, bin(POLYNOM)[2:])
    print(' ' * (len(bin(E_shift)) - len(bin(res[-1]))), bin(res[-1])[2:])

    W = bin(res[-1]).count('1')
    if W <= T:
        ok = True
        e = res[-1]
        print(f'W(R) = {W} <= {T}')
        break
    print(f'W(R) = {W} > {T}')
    E_shift = true_bin(E_shift, message_size)
    E_shift = int(E_shift[1:] + E_shift[0], 2)

if W == 0:
    print('There is no errors in this input')
    exit(0)
if not ok:
    print(f'cannot find error with this polnom {bin(POLYNOM)[2:]}, try another one')
    exit(1)
e_str = true_bin(e, POLY_SIZE)
e_true = (int(e_str + e_str, 2) >> i) % pow(2, POLY_SIZE)
E_true = E ^ e_true
I_true = int(true_bin(E_true, POLY_SIZE)[:-4][::-1], 2)
print()
print('e <- {:<2} = {}(2)'.format(i, true_bin(e, POLY_SIZE)))
print(f'e       = {true_bin(e_true, POLY_SIZE)}(2) = {oct(e_true)[2:]}(8)')
print(f'E_true  = E (xor) e = {true_bin(E_true, POLY_SIZE)}(2)')
print(f'I_true  = {bin(I_true)[2:]}(2) = {I_true}(10)')

I_true_array = list(map(int, true_bin(I_true, matr_size)))
E_matmul = int(''.join(map(str, matmul(I_true_array, matrix))), 2)
E1 = E_matmul ^ e_true
print()
print('CHECK')
print(f'E_matmul       = I x G = {true_bin(E_matmul, POLY_SIZE)}')
print(f'add error e    = {bin(e_true)[2:]}(2)')
print(f'E1 = E (xor) e = {true_bin(E1, POLY_SIZE)}')
print()
print(f'E        = {true_bin(E, POLY_SIZE)}')
print(f'E1       = {true_bin(E1, POLY_SIZE)}')
print(f'E_true   = {true_bin(E_true, POLY_SIZE)}')
print(f'E_matmul = {true_bin(E_matmul, POLY_SIZE)}')

if (E == E1 and E_true == E_matmul):
    print('~E == ~E1')
    print('E_true == E_matmul')
    print()
    print('Вектор ошибки и число найдены верно')
    print()
    print('-------------------------------------------------------')
    print('ANSWER')
    print(f'I_true  = {bin(I_true)[2:]}(2) = {I_true}(10)')
    print(f'e       = {true_bin(e_true, POLY_SIZE)}(2) = {oct(e_true)[2:]}(8)')
    print('-------------------------------------------------------')

else:
    print()
    print('Что-то пошло не так, мб числа другие попробуй, может в алгосе '
        'ошибка, хз')
print('gg wp ez')
