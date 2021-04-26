import pycurl
from io import BytesIO
import ast

# Loads database into test
bObj = BytesIO()
crl = pycurl.Curl()
crl.setopt(crl.URL,'https://t4g-dl-api.herokuapp.com/api/datasets/proj2/')
crl.setopt(crl.WRITEDATA, bObj)
crl.perform()
crl.close()
get_body = bObj.getvalue()
test = ast.literal_eval(get_body.decode('utf8'))

# Write data to data.txt
with open("data.txt", "w") as file:
    file.write(get_body.decode('utf8') + '\n')

