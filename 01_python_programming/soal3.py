# next_number_power_two

# input_number = int(input())
# i = 1
# while True:
#     next_number_power_two = 2**i
#     if next_number_power_two > input_number:
#         print(next_number_power_two)
#         break
#     i += 1


input_number = int(input())
p = 1
while p <= input_number:
    p *= 2
print(p)
