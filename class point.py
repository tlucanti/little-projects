import math
import builtins


class point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __del__(self):
		del self.x
		del self.y
		del self

	def __repr__(self):
		# return '<point object at {}>'.format(hex(id(self)))
		return self.__str__()

	def __str__(self):
		return '({}, {})'.format(str(self.x), str(self.y))

	def __bytes__(self):
		return bytes((self.x, self.y))

	def __copy__(self):
		return point(self.x, self.y)

	def __hash__(self):
		return hash(self.__str__())

	def __len__(self):
		return math.sqrt(self.x * self.x + self.y * self.y)

	def __getitem__(self, item):
		if item == 0 or item == -2:
			return self.x
		elif item == 1 or item == -1:
			return self.y
		else:
			raise IndexError('point has only two elements')

	def __setitem__(self, key, value):
		if key == 0 or key == -2:
			self.x = value
		elif key == 1 or key == -1:
			self.y = value
		else:
			raise IndexError('point has only two elements')

	def __reversed__(self):
		return point(self.y, self.x)

	def __contains__(self, item):
		return self.x == item or self.y == item

	def __bool__(self):
		return self.x != 0 and self.y != 0

	def __cmp__(self, other):
		if isinstance(other, list) or isinstance(other, tuple):
			other = point(other[0], other[1])
		return self.x * self.x + self.y * self.y - other.x * other.x + other.y * other.y

	def __lt__(self, other):
		if isinstance(other, list) or isinstance(other, tuple):
			other = point(other[0], other[1])
		return self.__abs__() < other.__abs__()

	def __le__(self, other):
		if isinstance(other, list) or isinstance(other, tuple):
			other = point(other[0], other[1])
		return self.__abs__() <= other.__abs__()

	def __eq__(self, other):
		if isinstance(other, point):
			return self.x == other.x and self.y == other.y
		elif isinstance(other, list) or isinstance(other, tuple):
			return self.x == other[0] and self.y == other[1]

	def __ne__(self, other):
		if isinstance(other, point):
			return self.x != other.x or self.y != other.y
		elif isinstance(other, list) or isinstance(other, tuple):
			return self.x != other[0] or self.y != other[1]

	def __gt__(self, other):
		if isinstance(other, list) or isinstance(other, tuple):
			other = point(other[0], other[1])
		return self.__abs__() > other.__abs__()

	def __ge__(self, other):
		if isinstance(other, list) or isinstance(other, tuple):
			other = point(other[0], other[1])
		return self.__abs__() >= other.__abs__()

	def __pos__(self):
		return self

	def __neg__(self):
		return point(-self.x, -self.y)

	def __abs__(self):
		return self.x * self.x + self.y * self.y

	def __invert__(self):
		return point(self.x, -self.y)

	def __round__(self, n=None):
		return point(round(self.x, n), round(self.y, n))

	def __floor__(self):
		return point(math.floor(self.x), math.floor(self.y))

	def __ceil__(self):
		return point(math.ceil(self.x), math.ceil(self.y))

	def __trunc__(self):
		return point(math.trunc(self.x), math.trunc(self.y))

	def __add__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x + other, self.y + other)
		elif isinstance(other, complex):
			return point(self.x + other.real, self.y + other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x + other[0], self.y + other[1])
		return point(self.x + other.x, self.y + other.y)

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x - other, self.y - other)
		elif isinstance(other, complex):
			return point(self.x - other.real, self.y - other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x - other[0], self.y - other[1])
		return point(self.x - other.x, self.y - other.y)

	def __rsub__(self, other):
		return self.__sub__(other)

	def __mul__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x * other, self.y * other)
		elif isinstance(other, complex):
			return point(self.x * other.real, self.y * other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x * other[0], self.y * other[1])
		return point(self.x * other.x, self.y * other.y)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __floordiv__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x // other, self.y // other)
		elif isinstance(other, complex):
			return point(self.x // other.real, self.y // other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x // other[0], self.y // other[1])
		return point(self.x // other.x, self.y // other.y)

	def __rfloordiv__(self, other):
		return self.__floordiv__(other)

	def __truediv__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x / other, self.y / other)
		elif isinstance(other, complex):
			return point(self.x / other.real, self.y / other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x / other[0], self.y / other[1])
		return point(self.x / other.x, self.y / other.y)

	def __rtruediv__(self, other):
		return self.__truediv__(other)

	def __mod__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			return point(self.x % other, self.y % other)
		elif isinstance(other, complex):
			return point(self.x % other.real, self.y % other.imag)
		elif isinstance(other, list) or isinstance(other, tuple):
			other = point(self.x % other[0], self.y % other[1])
		return point(self.x % other.x, self.y % other.y)

	def __rmod__(self, other):
		return self.__mod__(other)

	def __divmod__(self, other):
		return self.__floordiv__(other), self.__mod__(other)

	def __rdivmod__(self, other):
		return self.__divmod__(other)

	def __pow__(self, power, modulo=None):
		return point(pow(self.x, power, modulo), pow(self.y, power, modulo))

	def __lshift__(self, other):
		if other % 2:
			return self.__reversed__()
		else:
			return self

	def __rshift__(self, other):
		if other % 2:
			return self.__reversed__()
		else:
			return self

	def __iadd__(self, other):
		return self + other

	def __isub__(self, other):
		return self - other

	def __imul__(self, other):
		return self * other

	def __ifloordiv__(self, other):
		return self // other

	def __idiv__(self, other):
		return self / other

	def __itruediv__(self, other):
		return self / other

	def __imod__(self, other):
		return self % other

	def norm(self):
		ln = len(self)
		self.x /= ln
		self.y /= ln


def len(__n):
	if type(__n) == point:
		return __n.__len__()
	else:
		return builtins.len(__n)
