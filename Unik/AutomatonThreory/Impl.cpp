
#include <common.hpp>
#include <Impl.hpp>

#include <bit>

Impl::Impl(unsigned int num, bool inf)
	: num(num), p(0), pw(false), inf(inf)
{
	ind = std::popcount(num);
}

Impl::Impl(unsigned int num, unsigned int ind, unsigned int p, bool pw, bool inf)
	: num(num), ind(ind), p(p), pw(pw), inf(inf)
{}

bool Impl::operator==(const Impl &b)
{
	return num == b.num && ind == b.ind && p == b.p && pw == b.pw;
}

Impl::~Impl()
{}

bool Impl::can_patch(const Impl &a, const Impl &b)
{
	return a.num < b.num && a.p == b.p && b.ind - a.ind == 1 &&
	       std::popcount(b.num - a.num) == 1;
}

Impl Impl::patch(Impl &a, Impl &b) {
	a.pw = true;
	b.pw = true;

	return { a.num, a.ind, a.p + (b.num - a.num), false, a.inf or b.inf };
}

