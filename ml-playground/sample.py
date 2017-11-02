x = [4,50,3,10,88,39,2,15]

for i in range(0, len(x)):
    index = i
    for j in range(i+1, len(x)):
        if x[index] > x[j]:
            index = j
        j += 1
    if index != i:
        tmp = x[index]
        x[index] = x[i]
        x[i] = tmp
    i += 1
