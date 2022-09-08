import requests

#send request to API 2 to to synchronize data to firebase and save data excel file.
#the URL and the file name

url = 'http://127.0.0.1:5000/get_data'

file_name ='data_sent.csv'

data = "sync"
q="report"

x = requests.get(url, params={"data": data})
y = requests.get(url, params={"report": q})
