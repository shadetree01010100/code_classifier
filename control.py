import csv
import random


base_filename = 'stackoverflow.csv'
train_filename = 'train_' + base_filename
test_filename = 'test_' + base_filename

total = 0
yes = 0
no = 0
right = 0
wrong = 0
empty = 0
empty_code = 0

print('training set')
with open(train_filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        total += 1
        is_code = int(row[0])
        if is_code:
            yes += 1
        else:
            no += 1
        if not row[1]:
            if is_code:
                empty_code += 1
            else:
                empty += 1
        guess = random.randint(0, 1)
        if guess == is_code:
            right += 1
        else:
            wrong += 1

print('lines:', total)
print('random guess rate:', round(right / total * 100, 1), '%')
print('percent code:', round(yes / total * 100, 1), '%')
print('empty text lines:', round(empty / total * 100, 1), '%')
print('empty code lines:', round(empty_code / total * 100, 1), '%')
print()
total = 0
yes = 0
no = 0
right = 0
wrong = 0
empty = 0
empty_code = 0
print('test set')
with open(test_filename, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        total += 1
        is_code = int(row[0])
        if is_code:
            yes += 1
        else:
            no += 1
        if not row[1]:
            if is_code:
                empty_code += 1
            else:
                empty += 1
        guess = random.randint(0, 1)
        if guess == is_code:
            right += 1
        else:
            wrong += 1

print('lines:', total)
print('random guess rate:', round(right / total * 100, 1), '%')
print('percent code:', round(yes / total * 100, 1), '%')
print('empty text lines:', round(empty / total * 100, 1), '%')
print('empty code lines:', round(empty_code / total * 100, 1), '%')
