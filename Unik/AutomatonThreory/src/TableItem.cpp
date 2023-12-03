
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

bool TableItem::inf(void) const
{
	return inf_;
}

