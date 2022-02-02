# -*- coding: utf-8 -*-
# @Author: tlucanti
# @Date:   2022-01-30 18:20:34
# @Last Modified by:   tlucanti
# @Last Modified time: 2022-01-31 16:27:06

class IP:

	def __init__(self, ip=None):
		self.num = 0
		if ip is None:
			return
		if isinstance(ip, list) or isinstance(ip, tuple):
			if len(ip) != 4:
				raise IndexError('size of ip should be equal to 4')
			ip = f'{ip[0]}.{ip[1]}.{ip[2]}.{ip[3]}'
		if isinstance(ip, str):
			ip = list(map(int, ip.split('.')))
			self.num = ip[0]
			self.num *= 256
			self.num += ip[1]
			self.num *= 256
			self.num += ip[2]
			self.num *= 256
			self.num += ip[3]
		elif isinstance(ip, int):
			self.num = ip
		else:
			raise TypeError('excepted str, got ' + type(ip))

	def split(self):
		m = '{m:032b}'.format(m=self.num)
		ip = [m[:8], m[8:16], m[16:24], m[24:32]]
		ip = [int(ip[i], 2) for i in range(4)]
		# ip = list(map(int, ip))
		return ip

	def __str__(self):
		ip = self.split()
		return f'{ip[0]}.{ip[1]}.{ip[2]}.{ip[3]}'
		# return f'{bin(ip[0])}.{bin(ip[1])}.{bin(ip[2])}.{bin(ip[3])}'

	def set_mask(self, mask):
		self.__init__(int('{:0<32}'.format('1' * mask), 2))
		return self

	def __getitem__(self, idx):
		return self.split()[idx]

	def copy(self):
		return IP(self.num)


while True:
	line = input(' >> ')
	if line == 'q':
		break
	if '/' in line:
		ip, mask = line.split('/')
		mask = IP().set_mask(int(mask))
	else:
		ip, mask = line.split()
		mask = IP(mask)
	ip = IP(ip)
	net = IP([ip[i] & mask[i] for i in range(4)])
	start = net.copy()
	start.num += 1
	end = mask.copy()
	end.num = net.num + (end.num ^ 0xFFFFFFFF) - 1

	print(f'net  : {net}/{bin(mask.num).count("1")}')
	print('range:', start, '-', end)
		
