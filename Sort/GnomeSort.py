
def gnome_sort(arr):
    i = 0
    l = len(arr)
    while i < l:
        if i == 0:
            i += 1
        else:
            if arr[i - 1] <= arr[i]:
                i += 1
            else:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
                i -= 1
    return arr
