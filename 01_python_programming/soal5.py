# simple problem checker

input_string = input()


string_words_list = input_string.split()
wrong_words = list()

vowels = "aeiouAEIOU"


for word in string_words_list:
    if word.isupper():
        continue
    i = 0
    for char in word:
        if char.isalpha():
            if char not in vowels:
                i += 1
                if i == 5:
                    wrong_words.append(word)
                    break
            else:
                i = 0

print(" ".join(wrong_words))
