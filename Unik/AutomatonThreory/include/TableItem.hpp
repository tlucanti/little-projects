
#ifndef TABLE_ITEM_HPP
#define TABLE_ITEM_HPP

#include <common.hpp>

struct bitfield {
	bitfield(unsigned val, unsigned maxw)
		: val(val), maxw(maxw)
	{}

	unsigned operator [](unsigned i) const
	{
		if (i > maxw) {
			panic("out of range");
		}

		return (val >> i) & 0x1U;
	}

	unsigned val;
	unsigned maxw;
};

class TableItem {
public:
	TableItem(int val, int maxw);

	bitfield val;
};

#endif /* TABLE_ITEM_HPP */
