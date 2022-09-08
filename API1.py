from datetime import datetime
import os
from flask import Flask, request
import sqlite3

import re
from preprocessing import preproc
from model_1 import model

# removing special characters
def clean_symbols(text, specifics=[]):
    chars_to_clean = ['!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.','/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_','`', '{', '|', '}', '~', 'Â»', 'Â«', 'â€œ', 'â€']
    chars_to_split = ["'"]
    chars_to_clean.extend(specifics)
    punct_pattern = re.compile("[" + re.escape("".join(chars_to_clean)) + "]")
    text = re.sub(punct_pattern, "", text)
    split_pattern = re.compile("[" + re.escape("".join(chars_to_split)) + "]")
    text = re.sub(split_pattern, " ", text) 
    return text


'''
The API will receive the data from the user and it will preprocess the data (using preproc() function in preprocessing.py file) and predict
the data to detect bullying status(using the model() function in file model_1.py).

The API will connect to a sqlite3 database to create table (CyberBullying) and insert the data that predicted into it.
'''
app = Flask(__name__)
APP_ROUTE = os.path.dirname(os.path.abspath(__file__))

@app.route("/get_data", methods= ['GET'])
def get_data():
    data = request.args.get("data")
    id = request.args.get("id")
    
    #preprocess and predict
    data_pre = preproc(data)
    data_predicted=model(data_pre)
    #date of prediction
    date = datetime.now()

    if data_predicted==[1]:
        data_predicted='Offensive'
    else:
        data_predicted='Non-offensive'
    
    #Connect to the sqlite database
    conn = sqlite3.connect('informations.db')
    cur = conn.cursor()
    #Create the table
    
    query = '''CREATE TABLE IF NOT EXISTS CyberBullying(
                ID             INT          NOT NULL,
                TWEET          TEXT         NOT NULL,    
                DATE           timestamp    NOT NULL,
                DETECT         TEXT         NOT NULL
                );'''

    cur.execute(query)
    conn.commit()
    conn.close()
    

    # Insert the data into database 
    conn = sqlite3.connect('informations.db')
    cur = conn.cursor()
    cur.execute(f'INSERT INTO CyberBullying(ID,TWEET,DATE,DETECT) VALUES("{id}", "{clean_symbols(data)}", "{date}" , "{data_predicted}");')

    print ("Data inserted to the database")
    conn.commit()        
    conn.close()


    return 'Done'

if __name__ == "__main__":
    app.run(debug=False, host= '0.0.0.0', port= 5000)