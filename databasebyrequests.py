  
import requests

r = requests.get('https://t4g-dl-api.herokuapp.com/api/datasets/proj2/')
print(r.text)