
#ifndef TABLE_ITEM_HPP
#define TABLE_ITEM_HPP

#include <common.hpp>

struct bitfield {
	bitfield(unsigned val, unsigned maxw)
		: val(val), maxw(maxw)
	{}

	bitfield()
		: val(0), maxw(0)
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
	unsigned get(void) const;
	bool inf(void) const;

private:
	bitfield val;
	bool inf_;
};

inline std::ostream &operator <<(std::ostream &out, const TableItem &i)
{
	if (i.inf()) {
		out << '-';
	} else {
		out << i.get();
	}
	return out;
}

#endif /* TABLE_ITEM_HPP */
