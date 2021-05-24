import ast
import pickle

import numpy as np
from gensim.models import Doc2Vec

def split_data(dataset):
    # Split data
    test_data = []
    train_data = []

    for i, item in enumerate(dataset['data']):
        if i % TRAINING_RATIO == 0:
           test_data.append(item)
        else:
            train_data.append(item)
 
    return (test_data, train_data)


