import math

FAR = 1e10

def same_sign(a, b):
    return a == 0 or b == 0 or a * b >= 0

def different_sign(a, b):
    return a * b < 0

def intersect(v1, v2):
    A, B, C, D = v1.start, v1.end, v2.start, v2.end

    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1 * A.x + b1 * A.y

    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2 * C.x + b2 * C.y

    det = a1 * b2 - a2 * b1

    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return Point(x, y)

class Point():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f'({self.x}, {self.y})'


class Vector():
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def on_one_side(self, p1: Point, p2: Point):
        """
            return True if two points on the same side from line containing
            vector, or if one / both points are leaning on this line
        """
        s1 = self @ Vector(self.start, p1)
        s2 = self @ Vector(self.start, p2)
        return same_sign(s1, s2)

    def on_different_sides(self, p1: Point, p2: Point):
        """
            return True if two points on the different sides from line containing
            vector, and False if one / both points are leaning on this line
        """
        s1 = self @ Vector(self.start, p1)
        s2 = self @ Vector(self.start, p2)
        return different_sign(s1, s2)

    def __matmul__(self, other):
        a = self.end - self.start
        b = other.end - other.start
        return a.x * b.y - b.x * a.y

class Node():
    def is_lnode(self):
        pass

    def is_inode(self):
        pass

    def leftmost(self):
        pass

    def rightmost(self):
        pass

    def prev(self):
        pass

    def next(self):
        pass


class lnode(Node):
    def __init__(self, point: Point):
        self.point = point
        self.min = point
        self.max = point

    def is_lnode(self):
        return True

    def is_inode(self):
        return False

    def leftmost(self):
        return self.point

    def rightmost(self):
        return self.point

    def next(self, node):
        return None

    def prev(self, node):
        return None

    def find(self, point):
        return self.point

    def _repr(self, depth):
        return ' ' * depth + f'[{self.point.x}, {self.point.y}]'

    def __repr__(self):
        return self._repr(0)

class inode(Node):
    class Bridge():
        def __init__(self, left: Point, right: Point):
            self.br_left = left
            self.br_right = right

    def __init__(self, left: Node, right: Node, min: Point, max: Point, bridge: tuple):
        self.left = left
        self.right = right
        self.min = min
        self.max = max
        self.bridge = self.Bridge(bridge[0], bridge[1])

    def is_lnode(self):
        return False

    def is_inode(self):
        return True

    def leftmost(self):
        return self.left.leftmost()

    def rightmost(self):
        return self.right.rightmost()

    def next(self, point):
        br_left = self.bridge.br_left.x
        if point.x == br_left:
            return self.bridge.br_right
        elif point.x > br_left:
            return self.right.next(point)
        else:
            return self.left.next(point)

    def prev(self, point):
        br_right = self.bridge.br_right.x
        if point.x == br_right:
            return self.bridge.br_left
        elif point.x > br_right:
            return self.right.prev(point)
        else:
            return self.left.prev(point)

    def find(self, point):
        L = self.bridge.br_left.x
        R = self.bridge.br_right.x
        C = (L + R) / 2
        if point > R:
            return self.right.find(point)
        elif point > C:
            return self.bridge.br_right
        elif point >= L:
            return self.bridge.br_left
        else:
            return self.left.find(point)

    def _repr(self, depth):
        w = ' ' * depth
        s = w + f'bridge: [{self.bridge.br_left}, {self.bridge.br_right}]\n'
        s += self.left._repr(depth + 1) + '\n'
        s += self.right._repr(depth + 1) + '\n'
        return s

    def __repr__(self):
        return self._repr(0)


class segtree():
    def __init__(self, array: list):
        self.points = [Point(i, array[i]) for i in range(len(array))]
        self.tree = [None] * 4 * len(self.points)
        self._build(1, 0, len(self.points) - 1)

    def get_local_bridge(self, start, end, point: Point):
        self.subtree = [None] * len(self.points) * 4
        subtree_root = self._query(1, 0, len(self.points) - 1, start, end)
        return self._get_bridge(subtree_root, lnode(point))

    def _query(self, node, subtree_start, subtree_end, start, end):
        if start > end:
            return None
        assert start <= end
        if start == subtree_start and end == subtree_end:
            return self.tree[node]
        else:
            subtree_middle = (subtree_start + subtree_end) // 2
            left = self._query(node * 2, subtree_start, subtree_middle, start, min(end, subtree_middle))
            right = self._query(node * 2 + 1, subtree_middle + 1, subtree_end, max(start, subtree_middle + 1), end)

            if left is None:
                merged = right
            elif right is None:
                merged = left
            else:
                merged = self._merge(left, right)

            self.subtree[node] = merged
            return merged

    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = lnode(self.points[start])
        else:
            middle = (start + end) // 2
            self._build(node * 2, start, middle)
            self._build(node * 2 + 1, middle + 1, end)
            self.tree[node] = self._merge(self.tree[node * 2], self.tree[node * 2 + 1])

    def _merge(self, left, right) -> inode:
        if left.is_lnode() and right.is_lnode():
            return inode(left, right, left.point, right.point, [left.point, right.point])
        else:
            return inode(left, right, left.min, right.max, self._get_bridge(left, right))

    def _get_bridge_fini(self, L1, L2, L3, R1, R2, R3):
        L = L3 if L3 else L2
        R = R1 if R1 else R2

        R = self._highest_point(R1, R2, R3, L)
        L = self._highest_point(L1, L2, L3, R)
        R = self._highest_point(R1, R2, R3, L)
        L = self._highest_point(L1, L2, L3, R)
        return L, R

    def _highest_point(self, a, b, c, x):
        t1 = -FAR if a is None else math.atan2(a.y - x.y, abs(a.x - x.x))
        t2 = -FAR if b is None else math.atan2(b.y - x.y, abs(b.x - x.x))
        t3 = -FAR if c is None else math.atan2(c.y - x.y, abs(c.x - x.x))
        tm = max(t1, t2, t3)

        if tm == t2:
            return b
        elif tm == t1:
            return a
        else:
            return c

    def _get_bridge(self, left, right):
            Lmin = left.min.x
            Lmax = left.max.x

            Rmin = right.min.x
            Rmax = right.max.x

            while True:
                L = left.find((Lmin + Lmax) / 2)
                R = right.find((Rmin + Rmax) / 2)

                Lp = left.prev(L)
                if Lp is None:
                    Lp = Point(L.x, -FAR)
                Ln = left.next(L)
                if Ln is None:
                    Ln = Point(L.x, -FAR)

                Rp = right.prev(R)
                if Rp is None:
                    Rp =  Point(R.x, -FAR)
                Rn = right.next(R)
                if Rn is None:
                    Rn = Point(R.x, -FAR)

                if Lmax - Lmin <= Ln.x - Lp.x and Rmax - Rmin <= Rn.x - Rp.x:
                    Lp = left.prev(L)
                    Ln = left.next(L)
                    Rp = right.prev(R)
                    Rn = right.next(R)
                    return self._get_bridge_fini(Lp, L, Ln, Rp, R, Rn)
                if Lmax - Lmin <= Ln.x - Lp.x:
                    # left half is containing only 2 points: chose directly to prevent infinit loop
                    L = self._highest_point(left.prev(L), L, left.next(L), R)
                    Lp = left.prev(L)
                    if Lp is None:
                        Lp = Point(L.x, -FAR)
                    Ln = left.next(L)
                    if Ln is None:
                        Ln = Point(L.x, -FAR)
                if Rmax - Rmin <= Rn.x - Rp.x:
                    # right half in containing only 2 points: chose directly to prevent infinit loop
                    R = self._highest_point(right.prev(R), R, right.next(R), L)
                    Rp = right.prev(R)
                    if Rp is None:
                        Rp =  Point(R.x, -FAR)
                    Rn = right.next(R)
                    if Rn is None:
                        Rn = Point(R.x, -FAR)

                LR = Vector(L, R)

                Labove = True if left.is_lnode() else LR.on_one_side(Lp, Ln) # L is above it's neighbours
                Rabove = True if right.is_lnode() else LR.on_one_side(Rp, Rn) # R is above it's neighbours

                Linside = math.atan2(L.y - R.y, R.x - L.x) > math.atan2(Ln.y - R.y, R.x - Ln.x) # L is inside area under the bridge
                Rinside = math.atan2(R.y - L.y, R.x - L.x) < math.atan2(Rn.y - L.y, Rn.x - L.x) # R is inside area under the bridge

                if Labove and Rabove:
                    # case a: found the bridge
                    return (L, R)
                elif Rabove and Linside:
                    # case b: truncate left's right half and right's left half
                    Lmax = L.x
                    Rmin = R.x
                elif Rabove and not Linside:
                    # case c: truncate left's and right's left halves
                    Lmin = L.x
                    Rmin = R.x
                elif Labove and Rinside:
                    # case d: truncate left's right half and right's left half
                    Lmax = L.x
                    Rmin = R.x
                elif Labove and not Rinside:
                    # case e: truncate left's and right's right halves
                    Lmax = L.x
                    Rmax = R.x
                elif Linside and Rinside:
                    # case f: truncate left's right half and right's left half
                    Lmax = L.x
                    Rmin = R.x
                elif Linside and not Rinside:
                    # case g: trauncate only left's right half
                    Lmax = L.x
                elif Rinside and not Linside:
                    # case h: truncate only right's left half
                    Rmin = R.x
                else:
                    # case i:
                    assert not Rinside and not Linside and not Rabove and not Labove

                    mid = right.min.x
                    if intersect(Vector(L, Ln), Vector(R, Rp)).x <= mid:
                        # case i/1: truncate only left's left half
                        Lmin = L.x
                    else:
                        # case i/2: truncate only right's right half
                        Rmax = R.x

    def get_hc(self):
        return self._get_hc(self.tree[1])

    def _get_hc(self, root):
        point = root.leftmost()
        hc = []

        while point:
            hc.append((point.x, point.y))
            point = root.next(point)
        return hc

import numpy as np
def solve(array, window):
    array = np.array(array)
    out = np.zeros((2, len(array)))

    # upper convex hull
    st = segtree(array)

    out[0, 0] = 0
    out[0, 1] = math.atan2(array[0] - array[1], 1)
    for i in range(2, len(array)):
        bridge = st.get_local_bridge(max(0, i - 9), i - 1, Point(i, array[i]))
        ang = math.atan2(bridge[0].y - bridge[1].y, bridge[1].x - bridge[0].x)
        out[0, i] = ang

    # lower convex hull
    array *= -1
    st = segtree(array)

    out[1, 0] = 0
    out[1, 1] = out[0, 1]
    for i in range(2, len(array)):
        bridge = st.get_local_bridge(max(0, i - 9), i - 1, Point(i, array[i]))
        ang = math.atan2(bridge[1].y - bridge[0].y, bridge[1].x - bridge[0].x)
        out[1, i] = ang

    return out


def main():
    X = list(map(float, open('input.csv').readlines()))
    out = solve(X[:10000], 10)
    for i in range(out.shape[1]):
        print(f'{out[0, i]:g},{out[1, i]:g}')


def test():
    y = list(map(float, open('input.csv').readlines()))
    #y = [1, 0, 3, 0, 4, 0, 2, 0, 1, 0]
    y = [11384.96, 11373.01, 11324.0, 11329.41, 11289.0, 11275.09, 11259.0, 11258.34, 11277.64, 11285.74, 11272.63, 11256.93, 11246.78, 11287.5, 11277.02, 11298.6, 11296.81, 11292.39, 11290.0, 11306.58]
    y = [-i for i in y]
    v = 2
    st = segtree(y)
    print('Y:', st.points)
    print('HC:', st.get_hc())
    br = st.get_local_bridge(max(0, v - 9), v - 1, Point(v, y[v]))
    print(f'subtree {max(0, v - 9)} .. {v - 1}:', st._get_hc(st.subtree[1]))
    print(f'Bridge:', br)
    print(f'angle:', math.atan2(br[0].y - br[1].y, br[1].x - br[0].x))


if __name__ == '__main__':
    main()
