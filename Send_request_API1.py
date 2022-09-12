import requests
import pandas as pd

#send request to API to detect the data.
#the URL and the file name
url = 'http://127.0.0.1:5000/get_data'
file_name ='data_sent.csv'

#read the file
data = pd.read_csv(f'{file_name}', index_col=0)

#convert data to dictionary
data = data.to_dict()
data = data['full_text']

#send requests
for key, value in data.items():
    x = requests.post(url, params={"data": data[key], 'id':key})
