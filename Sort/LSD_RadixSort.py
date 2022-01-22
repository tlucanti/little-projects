import itertools

def lsd_radix_sort(arr):
	def get_digit(n, i):
		return n % ranks[i + 1] // ranks[i]
	dn = len(str(max(arr)))
	ranks = [1] * (dn + 1)
	for i in range(1, dn + 1):
		ranks[i] = ranks[i - 1] * 10
	for d in range(dn):
		bins = [[] for i in range(10)]
		for i in range(len(arr)):
			bins[get_digit(arr[i], d)].append(arr[i])
		arr = list(itertools.chain(*bins))
	return arr
