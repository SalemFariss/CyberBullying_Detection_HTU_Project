import os
from flask import Flask, request
import pandas as pd

import sqlite3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

'''
The API will receive requests to retrieve the data was predicted (report).
The API will connect to the database to synchronize it to cloud database (Firebase)
'''

app = Flask(__name__)
APP_ROUTE = os.path.dirname(os.path.abspath(__file__))


@app.route("/get_data", methods= ['GET'])
def get_data():
    r = request.args.get("data")
    q=request.args.get('report')
    if q=='report':
        conn = sqlite3.connect("informations.db")
        cur = conn.cursor()

        query="select * from CyberBullying"
        d=pd.read_sql(query,conn)
        # Saving the data to an excel sheet.
        d.to_csv('report.csv')

        conn.commit()
        conn.close()

    # logging to firebase
    if r=='sync':
        # Connect to the sqlite database
        conn = sqlite3.connect("informations.db")
        cur = conn.cursor()

        query="select * from CyberBullying"
        d=pd.read_sql(query,conn)
 
        # logging in using private key 
        cred = credentials.Certificate('project-htu-firebase-adminsdk-nzuna-2a1efd2349.json')
        firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-htu-default-rtdb.firebaseio.com/', 'httpTimeout' : 30})
        print('logged in to firebase')
    
        # Add the data to the database 
        for i in range(0, 29):
            # Date
            ref1 = f"{d.iloc[i, 0]}/DATE" 
            root = db.reference(ref1)
    
            # writing
            x = {'DATE': f'{d.iloc[i, 2]}'}
            root.set(x)
    
            # DETECT
            ref2 = f"{d.iloc[i, 0]}/DETECT"  
            root = db.reference(ref2)
    
            # writing
            x = {'DETECT': f'{d.iloc[i, 3]}'}
            root.set(x)

        conn.commit()
        conn.close()

    return 'Done'


if __name__ == "__main__":
    app.run(debug=False, host= '0.0.0.0', port= 5000)
