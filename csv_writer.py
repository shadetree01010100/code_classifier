import csv
import random

from stackoverflow import StackOverflowScraper


start_page = 1
num_pages = 1000
sos = StackOverflowScraper(2)
sos.page = start_page
base_filename = 'stackoverflow.csv'
train_filename = 'train_' + base_filename
test_filename = 'test_' + base_filename
train = 0
test = 0
i = 0
with open(train_filename, 'a', newline='', encoding='utf-8') as train_file, \
     open(test_filename, 'a', newline='', encoding='utf-8') as test_file:
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)
    for r in range(num_pages):
        print(
            'page: {}, total posts: {}'.format(r + 1, i),
            end='\r',
            flush=True)
        for post in sos.next_batch():
            i += 1
            if random.random() < 0.1:
                for line in post:
                    test_writer.writerow(line)
                test_writer.writerow([])
                test += 1
                continue
            for line in post:
                train_writer.writerow(line)
            train_writer.writerow([])
            train += 1
print('\n')
print('=' * 24)
print('train: {}, test: {}'.format(train, test))
