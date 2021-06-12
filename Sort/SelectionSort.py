
def selection_sort(arr):
    for i in range(len(arr)):
        mi = i
        for j in range(i, len(arr)):
            if arr[j] < arr[mi]:
                mi = j
        arr[mi], arr[i] = arr[i], arr[mi]
    return arr

print(selection_sort([5, 4, 3, 2, 1]))