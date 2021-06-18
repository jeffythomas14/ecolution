# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import joblib
import json

# Your API definition
app = Flask(__name__)


@app.route('/trail',methods=['GET'])
def trail():
	return jsonify({'Key': 'Hello World'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("json",data)
   
    x=[]
    for item in data:
        for data_item in item['Age']:
            for key in data_item:
                x.append(data_item[key])
    for item in data:
        for data_item in item['Location']:
            for key in data_item:
                x.append(data_item[key])            
            

    for item in data:
        for data_item in item['Occupation']:
            for key in data_item:
                x.append(data_item[key])

    result=15.3096*x[0] - 11.9896*x[1] + 3.4938*x[2] + 1.5125*x[3] + 5.3013*x[4] -6.5359*x[5] + 5.0965*x[6] + 12.84494*x[7] - 9.2372*x[8] + 4.6410*x[9] + 6.8138
    
            # query = pd.get_dummies(pd.DataFrame(json_))
            # print(query)
            # query = query.reindex(columns=model_columns, fill_value=0)

            # prediction = list(lr.predict(query))

    return jsonify({'prediction': abs(result)})

        

        
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 1234 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model1.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns1.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)