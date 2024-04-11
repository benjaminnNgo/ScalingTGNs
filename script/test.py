
test_list = [1,2,3,4,5]
index = [3,4]

test_shots = list(range(3,4))
result = [test_list[i] for i in index]
print(result)

result1 = [0]
for i in range(1,len(test_list)-1):
    result1.append(test_list[i])

print(result1)