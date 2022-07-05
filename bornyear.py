import csv

results = {}
with open('people.csv', newline='') as peoplefile:
    reader = csv.DictReader(peoplefile)
    for row in reader:
        results[row['Key']] = row['YrBorn']
print(results)
