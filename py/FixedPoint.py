##
#	Author:		kostya
#	Created:	2021-11-07 23:06:31
#	Modified:	2022-01-05 22:04:24
##

class FixedPoint()

	def __init__(self, val, exp, ready=False):
		self.exp = exp
		self.val = val * pow(10, (not ready) * exp)

	def __str__(self):
		lst = list(str(self.val)[::-1])
		lst.insert(exp, '.')
		return ''.join(lst)[::-1]

	def __add__(self, other):
		return FixedPoint(self.val + other.val, max(self.exp, other.exp), ready=True)

	def __sub__(self, other):
		return FixedPoint(self.val - other.val, max(self.exp, other.exp), ready=True)

	def __mul__(self, other):
		return FixedPoint(self.val * other.val, max(self.exp, other.exp), ready=True)

	def __truediv__(self, other):
		return FixedPoint(self.val // other.val, max(self.exp, other.exp), ready=True)


def sqrt(n):
	return sqrt()

def log():

def atan():


