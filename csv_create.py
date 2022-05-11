import csv
# file = open('data.csv',mode='w', newline='')

# writer = csv.writer(file)
# writer = csv.DictWriter(file, ['AR','top_x','top_y','label'])
# writer.writeheader()
# file.close()

file = open('data.csv',mode='a')
writer = csv.DictWriter(file, ['AR','top_x','top_y','label'])

writer.writerow({'AR':'1','top_x':2,'top_y':'3','label':'fall'})
file.close()