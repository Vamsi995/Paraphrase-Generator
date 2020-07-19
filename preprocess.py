# wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
# tar -xzf paws_wiki_labeled_final.tar.gz

import csv

train_examples = []
test_examples = []
dev_examples = []

with open("final/train.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")

    next(reader)

    for row in reader:

        if row[3] == "1":
            train_examples.append((row[1], row[2]))

with open("final/test.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")

    next(reader)

    for row in reader:

        if row[3] == "1":
            test_examples.append((row[1], row[2]))

with open("final/dev.tsv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")

    next(reader)

    for row in reader:

        if row[3] == "1":
            dev_examples.append((row[1], row[2]))

test_examples = dev_examples + test_examples


with open("PAW_Train.csv","w") as csvfile:
  writer = csv.writer(csvfile)

  for row in train_examples:
    writer.writerow(row)



with open("PAW_Test.csv","w") as csvfile:
  writer = csv.writer(csvfile)

  for row in test_examples:
    writer.writerow(row)
