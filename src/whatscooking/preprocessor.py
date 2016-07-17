__author__ = 'phx'

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

removeableWords = set(['frozen','fresh','freshly','dried','cooked','chopped',
                        'ground','grated','crushed','minced','cracked','shredded',
                        'large','medium','small','warm','plain','low-fat','low-sodium',
                        'fat-free','sweetened','bottled','low','sodium','refried','flaked',
                        'whole','and','canned','boneless','sliced','of','fat free','nonfat',
                        'reduced','steel-cut','gluten-free'
                    ])
wnl = WordNetLemmatizer()
def isMeaningful(word):
    return word not in removeableWords

def removeVerbFromIngredient(ingredient):
    from re import sub
    ings = filter(isMeaningful, sub('[^A-Za-z]+',' ',ingredient.lower()).split())
    ings = map(lemmatize,ings)

    return " ".join(ings)

def lemmatize(word):
    return wnl.lemmatize(word)


def cleanRecipe(recipe):
    recipe['ingredients'] =map(removeVerbFromIngredient,recipe['ingredients'])
    return recipe

def preprocess(data):
    data= map(cleanRecipe,data)
    return data

def getAttributeMap(train_data,combination_num):
    result = {}
    serial = 0
    for record in train_data:
        for ingredient in record['ingredients']:
            if ingredient not in result:
                result[ingredient] = serial
                serial+=1
    return result

def getLabelMap(train_data):
    result = {}
    serial = 0;
    for record in train_data:
        if record['cuisine'] not in result:
            result[record['cuisine']] = serial
            serial+=1
    return result

def getDataSet(train_data,attribute_map,label_map,hasLabel=True):
    attribute_size = len(attribute_map)
    record_num = len(train_data)
    from scipy.sparse import lil_matrix
    import numpy
    X = lil_matrix((record_num,attribute_size),dtype=numpy.int8)
    y_list =[]
    for i in xrange(0,record_num):
        record = train_data[i]
        if hasLabel:
            y_list.append(label_map.get(record['cuisine']))
        for ingredient in record['ingredients']:

            if attribute_map.get(ingredient):
                X[i,attribute_map.get(ingredient)]=1


    return X,numpy.array(y_list)