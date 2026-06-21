import math

input_number = int(input())

star_number = 1
for i in range(input_number, 0, -2):
    print(
        " " * math.floor(i / 2)
        + "*" * star_number
        + "*" * (star_number - 1)
        + 2 * (" " * math.floor(i / 2))
        + "*" * star_number
        + "*" * (star_number - 1)
    )
    star_number += 1


star_number = input_number
for j in range(1, input_number, 2):
    print(
        " " * math.ceil(j / 2)
        + "*" * math.floor(star_number / 2)
        + "*" * math.floor(star_number / 2 - 1)
        + 2 * (" " * math.ceil(j / 2))
        + "*" * math.floor(star_number / 2)
        + "*" * math.floor(star_number / 2 - 1)
    )
    star_number -= 2
