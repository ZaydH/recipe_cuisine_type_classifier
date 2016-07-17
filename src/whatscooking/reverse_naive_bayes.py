__author__ = 'phx'
import json
from preprocessor import preprocess,getLabelMap
train_file ="/Users/phx/downloads/competetion/recipe/train.json"
with open(train_file) as file:
     data = json.load(file)
print("size of dataset %d" % len(data))

data = preprocess(data)

train_data = [data[i] for i in xrange(0,len(data)) if i%3 !=0]
test_data = [data[i] for i in xrange(0,len(data)) if i%3 ==0]

label_map = getLabelMap(data)
def balanceData(data):
    categorized_data ={}
    for record in data:
        if not categorized_data.get(record['cuisine']):
            categorized_data[record['cuisine']] = [record]
        else:
            categorized_data[record['cuisine']].append(record)
    result = []
    for key,value in categorized_data.items():
        #print("%s : %d" % (key, len(value)))
        if len(value)<=300:
            result.extend(value)
        else:
            result.extend(value[:301])
    return result
train_data = balanceData(train_data)
print('label number : %d' %len(label_map))
print(label_map)

probs ={}
for record in train_data:
    for ingredient in record['ingredients']:
        if ingredient not in probs:
            prob = {'total':1}
            for k in label_map:
                prob[k] = 0
            prob[record['cuisine']] = 1
            probs[ingredient]=prob
        else:
            prob = probs[ingredient]
            prob['total'] = prob['total']+1
            prob[record['cuisine']] = prob[record['cuisine']]+1
    print(prob)
label_size = len(label_map)

for ingredient,prob in probs.items():
    for k in label_map:
        prob[k] = (prob[k]+1.0)/(prob['total']+label_size)

def predict(record):
    reverse_probs = {}
    for k in label_map:
        reverse_probs[k]=1
    for ingredient in record['ingredients']:
        if ingredient in probs:
            prob =probs[ingredient]
            for k in label_map:
                reverse_probs[k] = reverse_probs[k]*prob[k]
    max_prob = -1
    cuisine = ''
    for k,v in reverse_probs.items():
        if v > max_prob:
            cuisine=k

    print("%s to %s" % (record['cuisine'],cuisine))
    return cuisine

correct_num=0
for record in test_data:
    if predict(record) == record['cuisine']:
        correct_num +=1

print("accuracy is %0.3f%%" % (correct_num*1.0/len(test_data)))

