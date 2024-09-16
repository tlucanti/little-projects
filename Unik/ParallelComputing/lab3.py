
import numpy as np
import scipy
import time

def function(x):
	return 100 * x * np.log(x + 1) / (x + 100 * np.cos(0.1 * x) ** 2);

def main():
    ans = 0
    for i in range(1000):
        ans += scipy.integrate.quad(function, i * 1000, (i + 1) * 1000)[0]
    return ans

start = time.time()
ans = main()
end = time.time()

print(f'result: {ans}')
print(f'time: {end - start:.3f}s')

