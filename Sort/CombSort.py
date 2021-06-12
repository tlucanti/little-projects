
def comb_sort(arr):
    l = len(arr)
    falen = l
    while 1:
        falen = falen / 1.2473309
        alen = int(falen)
        for i in range(l - alen):
            if arr[i] > arr[i + alen]:
                arr[i], arr[i + alen] = arr[i + alen], arr[i]
        if alen <= 1:
            break
        for i in range(l - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1], = arr[i + 1], arr[i]
    return arr
