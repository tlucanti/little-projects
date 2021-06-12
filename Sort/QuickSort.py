
def qsort(arr, left=-1, right=-1):
    if left == -1:
        left = 0
        right = len(arr) - 1
    li = left
    ri = right
    pi = (left + right) // 2
    while ri - li > 0:
        while li < pi and arr[li] <= arr[pi]:
            li += 1
        while ri > pi and arr[ri] >= arr[pi]:
            ri -= 1
        arr[li], arr[ri] = arr[ri], arr[li]
        if pi == li:
            pi = ri
        elif pi == ri:
            pi = li
    if pi - left > 1:
        arr = qsort(arr, left, pi)
    if right - pi > 1:
        arr = qsort(arr, pi, right)
    return arr
