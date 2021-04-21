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
print(test) # To verify that everything worked, will spam output
