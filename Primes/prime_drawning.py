import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

def spin(arr, n, dtype=np.uint8):
	ans = np.zeros((n, n), dtype=dtype)
	if n % 2:
		i = j = n // 2
		global_counter = 0
		for size in range(n):
			sign = size % 2 * 2 - 1
			for i_it in range(size):
				ans[j][i] = arr[global_counter]
				global_counter += 1
				j += -sign
			for j_it in range(size):
				ans[j][i] = arr[global_counter]
				global_counter += 1
				i += sign
		for j_it in range(n - 1, -1, -1):
			ans[j_it][0] = arr[global_counter]
			global_counter += 1
	return ans


def reseto(size, start=0):
	i = 2
	res = np.ones(size + start, dtype=np.bool)
	while (i * i <= size):
		res[i + i::i] = 0
		i += 1
	res[0] = 0;
	res[1] = 0;
	return res[start:]


def draw(arr, size=(15, 15), save='PIL'):
	img = Image.fromarray(arr * 255)
	if save == 'PIL':
		img.show()
	elif save == True:
		img.save('./primes.jpg')
	else:
		# plt.figure(frameon=False)
		# plt.imshow(img, cmap=save)
		# plt.axis('off')
		#
		# fig = plt.figure(frameon=False)
		# fig.patch.set_visible(False)
		# ax.axis('off')
		# ax.imshow(img, cmap=save)
		plt.box(on=None)
		plt.show()


draw(reseto(1111 * 1111).reshape((1111, 1111)).astype(np.uint8), save=True)
draw(spin(reseto(1111 * 1111, 0), 1111), save=True)