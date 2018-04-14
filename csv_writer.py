import csv
import random

from stackoverflow import StackOverflowScraper


start_page = 1
num_pages = 10
sos = StackOverflowScraper()
sos.page = start_page
base_filename = 'stackoverflow.csv'
train_filename = 'train_' + base_filename
test_filename = 'test_' + base_filename
with open(train_filename, 'a', newline='', encoding='utf-8') as train_file, \
     open(test_filename, 'a', newline='', encoding='utf-8') as test_file:
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)
    for r in range(num_pages):
        for post in sos.next_batch():
            if random.random() < 0.1:
                for line in post:
                    test_writer.writerow(line)
            else:
                for line in post:
                    train_writer.writerow(line)
