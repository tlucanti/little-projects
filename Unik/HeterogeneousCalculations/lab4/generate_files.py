
import numpy as np

mat = (np.random.random([1000, 1000]) * 100).astype(int)
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

