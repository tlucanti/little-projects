import math

def same_sign(a, b):
    return a == 0 or b == 0 or a * b >= 0

def different_sign(a, b):
    return a * b < 0


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
        if p1 is None and p2 is None:
            return True

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

    def _repr(self, depth):
        return ' ' * depth + f'[{self.point.x}, {self.point.y}]'

    def __repr__(self):
        return self._repr(0)

class inode(Node):
    class Bridge():
        def __init__(self, left: Point, right: Point):
            self.br_left = left
            self.br_right = right

    def __init__(self, left: Node, right: Node, bridge: tuple):
        self.left = left
        self.right = right
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
            return inode(left, right, [left.point, right.point])
        else:
            return inode(left, right, self._get_bridge(left, right))

    def _get_bridge(self, left, right):
            L = left.rightmost()
            R = right.leftmost()

            Lp = left.prev(L)
            Ln = Point(L.x, -math.inf)

            Rp = Point(R.x, -math.inf)
            Rn = right.next(R)

            LR = Vector(L, R)
            while True:
                left_above = False
                right_above = False

                if left.is_lnode() or Lp is None:
                    left_above = True
                else:
                    left_above = LR.on_one_side(Lp, Ln)
                if right.is_lnode() or Rn is None:
                    right_above = True
                else:
                    right_above = LR.on_one_side(Rp, Rn)

                if left_above and right_above:
                    return (L, R)

                while Rn and LR.on_different_sides(Rp, Rn):
                    Rp = R
                    R = Rn
                    Rn = right.next(Rn)
                    LR = Vector(L, R)
                    if Rn is None:
                        break
                while Lp and LR.on_different_sides(Lp, Ln):
                    Ln = L
                    L = Lp
                    Lp = left.prev(Lp)
                    LR = Vector(L, R)
                    if Lp is None:
                        break

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
    st = segtree(array)
    out = np.zeros((2, len(array)))

    out[0, 0] = 0
    out[0, 1] = math.atan2(array[0] - array[1], 1)
    for i in range(2, 200):
        bridge = st.get_local_bridge(max(0, i - 9), i - 1, Point(i, array[i]))
        ang = math.atan2(bridge[0].y - bridge[1].y, bridge[1].x - bridge[0].x)
        out[0, i] = ang
    return out

def main():
    X = list(map(float, open('input.csv').readlines()))
    out = solve(X[:200], 10)
    for i in out[0]:
        print(format(i, 'g'))


def test():
    Y = list(map(float, open('input.csv').readlines()))
    y = [1, 0, 3, 0, 4, 0, 2, 0, 1, 0]
    st = segtree(y)
    print(st.points)
    print(st.get_hc())
    print(st.get_local_bridge(0, 2, Point(4, 4)))
    print(st._get_hc(st.subtree[1]))


if __name__ == '__main__':
    main()
