input_number = int(input())
factors = {}
d = 2
while input_number > 1:
    while input_number % d == 0:
        if d in factors:
            factors[d] += 1
        else:
            factors[d] = 1

        input_number //= d
    d += 1
    if d * d > input_number and input_number > 1:
        if input_number in factors:
            factors[input_number] += 1
        else:
            factors[input_number] = 1
        break

result = []
for k, v in factors.items():
    if v > 1:
        result.append(f"{k}^{v}")
    else:
        result.append(str(k))

print("*".join(result))
