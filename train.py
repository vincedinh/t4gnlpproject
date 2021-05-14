import ast
import pickle

import numpy as np
import multiprocessing
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

from read_data import *

# with open("data.txt") as file:
#     database = file.read()

# database = ast.literal_eval(database)
# test_data, train_data = split_data(database)

with open("train_data.pkl", "rb") as file:
    train_data = pickle.load(file)

with open("test_data.pkl", "rb") as file:
    test_data = pickle.load(file)

with open("categories.pkl", "rb") as file:
    categories = pickle.load(file)

with open("wordlist.pkl", "rb") as file:
    wordlist = pickle.load(file)

with open("weights.pkl", "rb") as file:
    weights = pickle.load(file)

# test_data, train_data = split_data
# wordlist = build_word_list(train_data)
# categories = build_category_vecs(wordlist, train_data)
# print(len(wordlist))

# training = gradient_descent(train_data, categories, wordlist, .001, 3, .001, weights)

#test_model(training[0], test_data, categories, wordlist)


# with open("weights.pkl", "wb") as file:
#     pickle.dump(training[0], file)

#constructs document vectors from models and document word list
def get_vectors(model, tagged_docs):
    sents = tagged_docs
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


cores = multiprocessing.cpu_count()
documents = [TaggedDocument(words=query["title"] + query["query"], tags=query["label"]) for query in train_data]
test_documents = [TaggedDocument(words=query["title"] + query["query"], tags=query["label"]) for query in test_data]
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores)
model_dbow.build_vocab([i for i in documents], keep_raw_vocab=False)


for epoch in range(10):
    model_dbow.train([i for i in tqdm(documents)], total_examples=len(documents), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

y_train, X_train = get_vectors(model_dbow, documents)
y_test, X_test = get_vectors(model_dbow, test_documents)

logreg = LogisticRegression(solver='saga', n_jobs=1, C=.01, max_iter=100, penalty='elasticnet', l1_ratio=.75)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(documents)], keep_raw_vocab=False)

for epoch in range(100):
    model_dmm.train([x for x in tqdm(documents)], total_examples=len(documents), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, X_train = get_vectors(model_dmm, documents)
y_test, X_test = get_vectors(model_dmm, test_documents)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))


new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

y_train, X_train = get_vectors(new_model, documents)
y_test, X_test = get_vectors(new_model, test_documents)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
