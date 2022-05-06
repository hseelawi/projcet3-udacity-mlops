import requests
import json

data = {'age': 50, 
        'workclass': 'Private', 
        'fnlgt': 367260,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married', 
        'occupation': 'Tech-support',
        'relationship': 'Unmarried', 
        'race': 'White', 
        'sex': 'Male',
        'capital-gain': 14084,
        'capital-loss': 0,
        'hours-per-week': 45,
        'native-country': 'Canada'}

url = 'https://udacity-mlop-project3-hs.herokuapp.com/predict'

resp = requests.post(url, data=json.dumps(data))
print(url)
print(json.dumps(data))
print(resp.status_code)
print(resp.text)
