import ast
import pickle

# Read in raw database
with open("data.txt") as file:
    database = file.read()

database = ast.literal_eval(database)

# Build word list from data
words = []
for item in database['data']:
    for key in item.keys():
        if key == '_id':
            continue
        else:
            for word in item[key]:
                if word not in words:
                    words.append(word)
words.sort()

# Export to pickle
with open("wordlist.pkl", "wb") as file:
    pickle.dump(words, file)

