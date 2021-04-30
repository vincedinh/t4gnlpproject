import ast
import pickle

# Read in raw database
with open("data.txt") as file:
    database = file.read()

database = ast.literal_eval(database)

def build_word_list():
    # Build word list from data
    words = []
    queries = 0
    for item in database['data']:
        for key in item.keys():
            if key == '_id':
                queries+=1
            else:
                for word in item[key]:
                    if word not in words:
                        words.append(word)
    words.sort()
    
    print("Queries:", queries)
    print("Items:", len(database['data']))
    # Export to pickle
    with open("wordlist.pkl", "wb") as file:
        pickle.dump(words, file)

build_word_list()
