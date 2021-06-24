import requests
import pandas as pd
import json
from connect_db import Interactive_db
import datetime


start = datetime.datetime.now()
###
# synonyms = 'https://tuna.thesaurus.com/relatedWords/'
word_path = 'word.csv'

word_ls = pd.read_csv(word_path, header=None)
# db = Interactive_db()

defines_ls = []
synonyms_ls = []
word_ls = word_ls.values[1:,]

for word in range(len(word_ls)):
    word_id = word_ls[word][1]
    print('word number = {}'.format(word), word_id)
    synonyms = 'https://tuna.thesaurus.com/relatedWords/' + word_id + '?limit=6'
    data = requests.get(synonyms)
    data = data.json()

    if (data['data']) is None:
        continue

    for i in range(len(data['data'])):
        # if defines_ls.count(data['data'][i]['definition']) != 0:
        #     define_id = defines_ls.index(data['data'][i]['definition']) + 1

        # else:
        defines_ls.append([word+1, data['data'][i]['definition']])
        # define_id = len(defines_ls)
        # print(data['data'][define_id]['definition'])

        for j in range(len(data['data'][i]['synonyms'])):
            # print(data['data'][i]['synonyms'][j]['term'])
            synonyms_ls.append([word+1, len(defines_ls), data['data'][i]['synonyms'][j]['term'],
                                data['data'][i]['synonyms'][0]['similarity']])


db = Interactive_db()
# add define table
sql = 'INSERT INTO define(idword, define) VALUE (%s, %s)'
tupleA = tuple(defines_ls)
print(tupleA)   # 16726 define of 3000 word
db.insert_list(sql, tupleA)


sql1 = 'INSERT INTO synonyms(idword, iddefine , synonyms, similarity) VALUE (%s, %s, %s, %s)'
tupleB = tuple(synonyms_ls)
print(tupleB)
db.insert_list(sql1, tupleB)  # 371301 synonyms of 3000 word

defines_ls = pd.DataFrame(defines_ls, columns=['word_id', 'definition'])
synonyms_ls = pd.DataFrame(synonyms_ls, columns=['word_id','define_id', 'synonym', 'similarity'])
defines_ls.to_csv('definitions.csv')
synonyms_ls.to_csv('synonyms.csv')

end = datetime.datetime.now()

time = end - start
print('---------------------Time line--= {}'.format(time))
# ---------------------Time line--= 0:21:21.898879
break_point = 'stop debug'

print('end program')
