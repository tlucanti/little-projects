
# ------------------------------------------------------------------------------
POLY_SIZE   = 15        # polynom size: x^15
POLYNOM     = 0b010011  # binary polynom: x5-x4-x3-x2-x1-x0
N           = 1600      # message cnt
I           = 1000        # message data
e           = 0o1000      # octal error
T           = 1         # error count
# ------------------------------------------------------------------------------

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
    if len(residues) == 0:
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
print(f'POLYNOM  = {bin(POLYNOM)[2:]}')
print(f'N        = {N}')
print(f'I        = {bin(I)}(2) = {I}')
print('e        = {e:o}(8) = {e:b}(2)'.format(e=e))
print('-------------------------------------------------------')

print()
print("2 ^ R >= Ne + 1")
print("2 ^ R >= R + K + T")
__e_inp = e
K = 0
while pow(2, K) < N:
    K += 1
print(f"K = ceil(log2(N)) = {K}")
R = 0
while True:
    if pow(2, R) < R + K + T:
        op = '<'
    else:
        op = '>='
    print(f'{pow(2, R)} = 2 ^ {R} {op} {R} + {K} + {T} = {R + K + T}')
    if op == '>=':
        break
    R += 1

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

I_str = '{data:0{size}b}'.format(data=I, size=matr_size)
I_arr = list(map(int, I_str))
E_arr = matmul(I_arr, matrix)
E_str = ''.join(map(str, E_arr))
E = int(E_str, 2)
E_err = E ^ e
E_err_str = '{E:0{size}b}'.format(E=E_err, size=matr_size)
e_str = '{e:0{size}b}'.format(e=e, size=matr_size)

print()
print(f'E = I x G = {I_str} x G = {E_str}')
print(f'E_err = E (xor) e = {E_str} (xor) {e_str} = {E_err_str}')

E_shift = E_err
message_size = len(matrix[0])
ok = False
for i in range(15):
    print()
    print(f'E << {i}')
    res, sh = xor_div(E_shift, POLYNOM)

    space = 0
    print(f'_{bin(E_shift)[2:]}|{bin(POLYNOM)[2:]}')
    for j in range(len(res) - 1):
        print(' ' * space, bin(POLYNOM)[2:])
        space += sh[j]
        print(' ' * space, '_', bin(res[j])[2:], sep='')
    if len(res) > 1:
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
E_true = E_err ^ e_true
I_true = int(true_bin(E_true, POLY_SIZE)[::-1], 2)
print()
print('e <- {:<2} = {}(2)'.format(i, true_bin(e, POLY_SIZE)))
print(f'e       = {true_bin(e_true, POLY_SIZE)}(2) = {oct(e_true)[2:]}(8)')
print(f'E_true  = E_err (xor) e = {true_bin(E_true, POLY_SIZE)}(2)')
print(f'I_true  = {bin(I_true)[2:]}(2) = {I_true}(10)')

print(e, e_true, I, I_true)
if (__e_inp == e_true and I_true == I):
    print()
    print('Исправить ошибку удалось')
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
