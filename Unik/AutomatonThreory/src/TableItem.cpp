
#include <TableItem.hpp>

TableItem::TableItem(int val, int maxw)
	: val()
{
	if (val >= 0) {
		this->val = bitfield(val, maxw);
	}
	inf_ = (val < 0);
}

unsigned TableItem::get(void) const
{
	if (inf()) {
		panic("getting value of inf item");
	}
	return val.val;
}

std::string TableItem::bin(int maxw) const
{
	unsigned v = val.val;
	std::string ret(maxw, '0');
	unsigned i = maxw - 1;

	if (inf()) {
		ret = std::string(maxw, '-');
		return ret;
	}

	while (v) {
		ret.at(i) = '0' + (v % 2);
		v /= 2;
		--i;
	}
	return ret;
}


bool TableItem::get_bit(int i) const
{
	return val[i];
}

bool TableItem::inf(void) const
{
	return inf_;
}

int TableItem::get_w(void) const
{
	return val.maxw;
}

