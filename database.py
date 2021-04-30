import pycurl
import pickle
from io import BytesIO
import ast

TRAINING_SIZE = 16798

# Loads database into test
bObj = BytesIO()
crl = pycurl.Curl()
crl.setopt(crl.URL,'https://t4g-dl-api.herokuapp.com/api/datasets/proj2/')
crl.setopt(crl.WRITEDATA, bObj)
crl.perform()
crl.close()
get_body = bObj.getvalue()
full_data = ast.literal_eval(get_body.decode('utf8'))


# Split data
test_data = []
train_data = []

for i, item in enumerate(full_data['data']):
    if i < TRAINING_SIZE:
        train_data.append(item)
    else:
        test_data.append(item)
 
# Write to file
with open("train_data.pkl", "wb") as file:
    pickle.dump(train_data, file)

with open("test_data.pkl", "wb") as file:
    pickle.dump(test_data, file)
