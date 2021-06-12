
def odd_even_sort(arr):
    sorted_ = False
    while not sorted_:
        sorted_ = True
        for i in range(0, len(arr) - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted_ = False
        for i in range(1, len(arr) - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted_ = False
    return arr
