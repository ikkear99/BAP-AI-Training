from connect_db import Interactive_db
import pandas as pd

word_ls = pd.read_csv('word.csv')
word = input("keywords to search: ")

word_ls = list (word_ls.values[:,1])

if word_ls.count(word) == 1:
    db = Interactive_db()
    sql = "select  word, define, synonyms as synonym, similarity from word as w inner join define as df on w.idword = df.idword inner join synonyms as sy on df.iddefine = sy.iddefine WHERE word = '" + word + "'"
    key = db.select(sql)
    key = pd.DataFrame(key, columns=['word', 'define', 'synonyms', 'similarity '])
    key.to_csv("Synonyms-" + word + ".csv")
else:
    print("No data!")
