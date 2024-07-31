
from num import numbers
import number_parser
import sys

def hamming(a, b):
    assert len(a) == len(b)

    res = 0
    for c1, c2 in zip(a, b):
        res += c1 != c2

    return res / len(a)


def find_nearest(w):
    closest = None
    dist = len(w)

    if w in numbers:
        # print(f'found: {w} = {numbers[w]}')
        return w

    for num in numbers:
        if len(num) != len(w):
            continue

        ham = hamming(num, w)
        if ham < dist:
            dist = ham
            closest = num

    # print(f'found: {w} = {numbers[closest]} (dist={dist})')
    assert dist <= 1 / len(w)
    return closest


def parse(s):
    fixed = ' '.join(find_nearest(w) for w in s.split())
    res = number_parser.parse_number(fixed)

    assert res is not None
    return res


def solve(nums):
    ans = 0
    n = []

    for i in range(1, len(nums) - 1):
        dist = abs(nums[i + 1] - nums[i - 1])
        if dist > ans:
            ans = dist
            n = [nums[i - 1], nums[i + 1]]

    return ans, n


def main():
    txt = sys.stdin.read()

    numbers = [parse(s) for s in txt.split(',')]
    # print(numbers)
    print(solve(numbers))


if __name__ == '__main__':
    main()

