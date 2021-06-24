import requests
import json
import pandas as pd
from connect_db import Interactive_db

URL = 'https://www.ef-australia.com.au/assetscdn/WIBIwq6RdJvcD9bc8RMd/cefcom-english-resources-site/page-data/au/english-resources/english-vocabulary/top-3000-words/page-data.json?v=796d67c48b3a2d5ac497b2c9ba950275'

page = requests.get(URL)

data = page.json()

with open('data_word.json','w') as fp:
    json.dump(data, fp, indent=4)

dataList = []
li_key = []
jsonPath = 'data_word.json'

with open(jsonPath, 'r') as fp:
    data_page = json.load(fp)
    key = data_page.get('result').get('pageContext').get('data').get('page').get('mainContent')

    li_key = key.split('</p>')
    li_key = li_key[1].split('<br>')
    del li_key[0]

    for i in range(len(li_key)):
        li_key[i] = li_key[i].strip()

    df = pd.DataFrame(li_key, columns=['keys'])
    df.to_csv('keys.csv')

word_path = 'word.csv'
synonyms_ls = []
define_ls = []

db = Interactive_db()
word_ls = pd.read_csv(word_path, header=None)

i = 0
for i in range(len(word_ls)):
    sql = "INSERT INTO word(idword, word) VALUE (%s, %s)"
    param = (i+1 ,word_ls[1][i+1])
    db.insert(sql, param)
    print(i)
