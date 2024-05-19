
import numpy as np

mat = np.random.randint(0, 10, 1000 * 1000)
mat = mat.reshape((1000, 1000))
print('generated')

def to_str(arr):
    return '\n'.join(map(str, arr.flatten().tolist()))

with open('matrix.txt', 'w') as f:
    f.write(to_str(mat))
    print('matrix.txt done')

with open('result_rows.txt', 'w') as f:
    f.write(to_str(np.average(mat, axis=1)))
    print('result_rows.txt done')

with open('result_cols.txt', 'w') as f:
    f.write(to_str(np.average(mat, axis=0)))
    print('result_cols.txt done')

