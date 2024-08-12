
#include <cmath>
#include <iostream>
#include <stack>
#include <vector>

static constexpr double FAR = 1e10;

struct Point;
struct Vector;

static bool same_sign(double a, double b);
static bool different_sign(double a, double b);
static Point intersect(const Vector &v1, const Vector &v2);

struct Point {
	int x;
	double y;

	Point(int x, double y)
		: x(x), y(y)
	{}

	Point()
		: x(0), y(0)
	{}

	Point operator -(const Point &other) const
	{
		return Point(x - other.x, y - other.y);
	}
};

struct Vector {
	Point start;
	Point end;

	Vector(const Point &start, const Point &end)
		: start(start), end(end)
	{}

	double operator *(const Vector &other) const
	{
		const Point a = end - start;
		const Point b = other.end - other.start;
		return a.x * b.y - b.x * a.y;
	}

	bool on_same_side(const Point &p1, const Point &p2) const
	{
		const double s1 = *this * Vector(start, p1);
		const double s2 = *this * Vector(start, p2);
		return same_sign(s1, s2);
	}

	bool on_different_sides(const Point &p1, const Point &p2)
	{
		const double s1 = *this * Vector(start, p1);
		const double s2 = *this * Vector(start, p2);
		return different_sign(s1, s2);
	}
};

static inline bool same_sign(double a, double b)
{
	return a * b >= 0;
}

static inline bool different_sign(double a, double b)
{
	return a * b < 0;
}

static inline Point intersect(const Vector &v1, const Vector &v2)
{
	const Point A = v1.start,
		    B = v1.end,
		    C = v2.start,
		    D = v2.end;

	const double a1 = B.y - A.y;
	const double b1 = A.x - B.x;
	const double c1 = a1 * A.x + b1 * A.y;

	const double a2 = D.y - C.y;
	const double b2 = C.x - D.x;
	const double c2 = a2 * C.x + b2 * C.y;

	const double det = a1 * b2 - a2 * b1;

	const double x = (b2 * c1 - b1 * c2) / det;
	const double y = (a1 * c2 - a2 * c1) / det;
	return Point(x, y);
}


struct segtree {
	struct Node {
		virtual bool is_lnode(void) const = 0;
		virtual bool is_inode(void) const = 0;
		virtual const Point *prev(const Point &point) const = 0;
		virtual const Point *next(const Point &point) const = 0;
		virtual Point max(void) const = 0;
		virtual Point min(void) const = 0;
		virtual Point find(double point) const = 0;
		virtual Point find(int point) const = 0;
		virtual ~Node(void) {}
	};

	struct Lnode : public Node {
		Point point;

		Lnode(const Point &point)
			: point(point)
		{}

		bool is_lnode(void) const override
		{
			return true;
		}

		bool is_inode(void) const override
		{
			return false;
		}

		Point *prev(const Point &p) const override
		{
			(void)p;
			return nullptr;
		}

		Point *next(const Point &p) const override
		{
			(void)p;
			return nullptr;
		}

		Point max(void) const override
		{
			return point;
		}

		Point min(void) const override
		{
			return point;
		}

		Point find(double p) const override
		{
			(void)p;
			return point;
		}

		 Point find(int p) const override
		 {
			 (void)p;
			 return point;
		 }
	};

	struct Inode : public Node {
		struct Bridge {
			Point br_left;
			Point br_right;

			Bridge(const Point &left, const Point &right)
				: br_left(left), br_right(right)
			{}
		};

		const Node *left;
		const Node *right;
		Point _min;
		Point _max;
		Bridge bridge;

		Inode(const Node *left, const Node *right,
		      const Point &min, const Point &max,
		      const std::pair<Point, Point> &bridge)
			: left(left), right(right), _min(min), _max(max),
			  bridge(bridge.first, bridge.second)
		{}

		bool is_lnode(void) const override
		{
			return false;
		}

		bool is_inode(void) const override
		{
			return true;
		}

		const Point *next(const Point &point) const override
		{
			int br_left = bridge.br_left.x;

			if (point.x == br_left)
				return &bridge.br_right;
			else if (point.x > br_left)
				return right->next(point);
			else
				return left->next(point);
		}

		const Point *prev(const Point &point) const override
		{
			int br_right = bridge.br_right.x;

			if (point.x == br_right)
				return &bridge.br_left;
			else if (point.x > br_right)
				return right->prev(point);
			else
				return left->prev(point);
		}

		Point min(void) const override
		{
			return _min;
		}

		Point max(void) const override
		{
			return _max;
		}

		Point find(double point) const override
		{
			int L = bridge.br_left.x;
			int R = bridge.br_right.x;
			double C = (L + R) / 2.0;

			if (point > R)
				return right->find(point);
			else if (point > C)
				return bridge.br_right;
			else if (point >= L)
				return bridge.br_left;
			else
				return left->find(point);
		}

		Point find(int point) const override
		{
			int L = bridge.br_left.x;
			int R = bridge.br_right.x;
			double C = (L + R) / 2.0;

			if (point > R)
				return right->find(point);
			else if (point > C)
				return bridge.br_right;
			else if (point >= L)
				return bridge.br_left;
			else
				return left->find(point);
		}
	};

	const std::vector<double> &array;
	std::vector<Node *> tree;
	std::stack<Node *> freelist;

	segtree(const std::vector<double> &array)
		: array(array)
	{
		tree.resize(4 * array.size());
		_build(1, 0, array.size() - 1);
	}

	~segtree(void)
	{
		for (Node *node : tree) {
			delete node;
		}
	}

	std::pair<Point, Point> query_bridge(int start, int end, const Point &point)
	{
		Node *subtree = _query(1, 0, array.size() - 1, start, end);
		Lnode right = Lnode(point);
		auto ret = _get_bridge(subtree, &right);

		while (not freelist.empty()) {
			delete freelist.top();
			freelist.pop();
		}

		return ret;
	}

	Node *_query(int node, int subtree_start, int subtree_end, int start, int end)
	{
		if (start > end)
			return nullptr;

		if (start == subtree_start and end == subtree_end) {
			return tree.at(node);
		} else {
			int subtree_middle = (subtree_start + subtree_end) / 2;
			Node *left = _query(node * 2, subtree_start, subtree_middle,
					    start, std::min(end, subtree_middle));
			Node *right = _query(node * 2 + 1, subtree_middle + 1, subtree_end,
					     std::max(start, subtree_middle + 1), end);
			Node *merged;

			if (left == nullptr) {
				merged = right;
			} else if (right == nullptr) {
				merged = left;
			} else {
				merged = _merge(left, right);
				freelist.push(merged);
			}
			return merged;
		}
	}

	void _build(int node, int start, int end)
	{
		if (start == end) {
			tree.at(node) = new Lnode(Point(start, array.at(start)));
		} else {
			int middle = (start + end) / 2;
			_build(node * 2, start, middle);
			_build(node * 2 + 1, middle + 1, end);
			tree.at(node) = _merge(tree.at(node * 2), tree.at(node * 2 + 1));
		}
	}

	Node *_merge(const Node *left, const Node *right)
	{
		std::pair<Point, Point> bridge;

		if (left->is_lnode() and right->is_lnode()) {
			bridge = { left->min(), right->max() };
		} else {
			bridge = _get_bridge(left, right);
		}

		return new Inode(left, right, left->min(), right->max(), bridge);
	}

	std::pair<Point, Point>
	__get_bridge_fini(const Point *L1, const Point *L2, const Point *L3,
			  const Point *R1, const Point *R2, const Point *R3)
	{
		Point L = L3 ? *L3 : *L2;
		Point R = R1 ? *R1 : *R2;

		R = __highest_point(R1, R2, R3, L);
		L = __highest_point(L1, L2, L3, R);
		R = __highest_point(R1, R2, R3, L);
		L = __highest_point(L1, L2, L3, R);

		return { L, R };
	}

	Point __highest_point(const Point *a, const Point *b, const Point *c, Point x)
	{
        	const double t1 = a ? atan2(a->y - x.y, abs(a->x - x.x)) : -FAR;
        	const double t2 = b ? atan2(b->y - x.y, abs(b->x - x.x)) : -FAR;
		const double t3 = c ? atan2(c->y - x.y, abs(c->x - x.x)) : -FAR;

		if (t2 >= t1 and t2 >= t3)
			return *b;
		else if (t1 >= t2 and t1 >= t3)
			return *a;
		else
			return *c;
	}

	Point __safe_prev(const Node *node, const Point &pos, const Point &other)
	{
		const Point *p = node->prev(pos);

		if (p == nullptr)
			return other;
		else
			return *p;
	}

	Point __safe_next(const Node *node, const Point &pos, const Point &other)
	{
		const Point *p = node->next(pos);

		if (p == nullptr)
			return other;
		else
			return *p;
	}

	std::pair<Point, Point> _get_bridge(const Node *left, const Node *right)
	{
		int Lmin = left->min().x;
		int Lmax = left->max().x;

		int Rmin = right->min().x;
		int Rmax = right->max().x;

		while (true) {
			Point L = left->find((Lmin + Lmax) / 2.0);
			Point R = right->find((Rmin + Rmax) / 2.0);

			Point Lp = __safe_prev(left, L, Point(L.x, -FAR));
			Point Ln = __safe_next(left, L, Point(L.x, -FAR));

			Point Rp = __safe_prev(right, R, Point(R.x, -FAR));
			Point Rn = __safe_next(right, R, Point(R.x, -FAR));

			if (Lmax - Lmin <= Ln.x - Lp.x and Rmax - Rmin <= Rn.x - Rp.x) {
				return __get_bridge_fini(
						left->prev(L), &L, left->next(L),
						right->prev(R), &R, right->next(R));
			}
			if (Lmax - Lmin <= Ln.x - Lp.x) {
                    		// left half is containing only 2 points,
				// chose directly to prevent infinit loop
				L = __highest_point(left->prev(L), &L, left->next(L), R);
				Lp = __safe_prev(left, L, Point(L.x, -FAR));
				Ln = __safe_next(left, L, Point(L.x, -FAR));
			}
			if (Rmax - Rmin <= Rn.x - Rp.x) {
                    		// right half in containing only 2 points,
				// chose directly to prevent infinit loop
				R = __highest_point(right->prev(R), &R, right->next(R), L);
				Rp = __safe_prev(right, R, Point(R.x, -FAR));
				Rn = __safe_next(right, R, Point(R.x, -FAR));
			}

			Vector LR = Vector(L, R);

			bool Labove = left->is_lnode() ? true : LR.on_same_side(Lp, Ln);
			bool Rabove = right->is_lnode() ? true : LR.on_same_side(Rp, Rn);

                	bool Linside = atan2(L.y - R.y, R.x - L.x) > atan2(Ln.y - R.y, R.x - Ln.x);
                	bool Rinside = atan2(R.y - L.y, R.x - L.x) < atan2(Rn.y - L.y, Rn.x - L.x);

			if (Labove and Rabove) {
				// case a: found the bridge
				return { L, R };
			} else if (Rabove and Linside) {
				// case b: truncate left's right half and right's left half
				Lmax = L.x;
				Rmin = R.x;
			} else if (Rabove and not Linside) {
				// case c: truncate left's and right's left halves
				Lmin = L.x;
				Rmin = R.x;
			} else if (Labove and Rinside) {
				// case d: truncate left's right half and right's left half
				Lmax = L.x;
				Rmin = R.x;
			} else if (Labove and not Rinside) {
				// case e: truncate left's and right's right halves
				Lmax = L.x;
				Rmax = R.x;
			} else if (Linside and Rinside) {
				// case f: truncate left's right half and right's left half
				Lmax = L.x;
				Rmin = R.x;
			} else if (Linside and not Rinside) {
				// case g: trauncate only left's right half
				Lmax = L.x;
			} else if (Rinside and not Linside) {
				// case h: truncate only right's left half
				Rmin = R.x;
			} else {
				// case i:

				int mid = right->min().x;
				if (intersect(Vector(L, Ln), Vector(R, Rp)).x <= mid) {
					// case i/1: truncate only left's left half
					Lmin = L.x;
				} else {
					// case i/2: truncate only right's right half
					Rmax = R.x;
				}
			}
		}
	}
};

#ifndef SIZE
# define SIZE 100000
#endif

#ifndef WINDOW
# define WINDOW 10
#endif

struct Angles {
	double a;
	double b;
};

static void calc(std::vector<double> &inputs, std::vector<Angles> &out)
{
	// upper convex hull
	segtree upper = segtree(inputs);

	out.at(0).a = 0;
	out.at(1).a = atan2(inputs.at(0) - inputs.at(1), 1);
	for (int i = 2; i < (int)inputs.size(); i++) {
        	auto bridge = upper.query_bridge(std::max(0, i - WINDOW + 1), i - 1, Point(i, inputs.at(i)));
        	out.at(i).a = atan2(bridge.first.y - bridge.second.y, bridge.second.x - bridge.first.x);
	}

	// lower convex hull
	for (double &i : inputs) {
		i *= -1;
	}
	segtree lower = segtree(inputs);

	out.at(0).b = 0;
	out.at(1).b = out.at(1).a;
	for (int i = 2; i < (int)inputs.size(); i++) {
        	auto bridge = lower.query_bridge(std::max(0, i - WINDOW + 1), i - 1, Point(i, inputs.at(i)));
        	out.at(i).b = atan2(bridge.second.y - bridge.first.y, bridge.second.x - bridge.first.x);
	}
}

int main()
{
	std::vector<double> inputs(SIZE);
	std::vector<Angles> outputs(SIZE);

	for (int i = 0; i < SIZE; i++) {
		std::cin >> inputs.at(i);
	}

	calc(inputs, outputs);

	for (const auto &ang : outputs) {
		std::cout << ang.a << ',' << ang.b << '\n';
	}
}

