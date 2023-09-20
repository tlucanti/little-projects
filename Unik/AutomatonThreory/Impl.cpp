
#include <common.h>
#include <Impl.hpp>

#include <bit>

Impl::Impl(unsigned int num, bool inf)
	: num(num), p(0), p2(false), inf(inf)
{
	ind = std::popcount(num);
}

Impl::~Impl()
{}

bool Impl::can_patch(const Impl &a, const &Impl b)
{
	return a.num < b.num && a.p == b.p && b.ind - a.ind == 1 &&
	       std::popcount(b.num - a.num) == 1;
}

Impl Impl::patch(Impl &a, Impl &b) {
	a.pw = true;
	b.pw = true;

	return {.num = a.num,
		.ind = a.ind,
		.p = a.p + (b.num - a.num),
		.pw = false,
		.inf = a.inf or b.inf};
}

