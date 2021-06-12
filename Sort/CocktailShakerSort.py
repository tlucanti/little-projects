
def shaker_sort(arr):
    beg = 0
    end = len(arr)
    while beg < end:
        for i in range(beg, end - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        for i in range(end - 1, beg, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
        beg += 1
        end -= 1
    return arr
