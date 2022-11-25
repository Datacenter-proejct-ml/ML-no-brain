from flask import Flask, request, Response, send_file
import jsonpickle
import pandas as pd

# app = Flask(__name__)

# @app.route('/api/preprocess/<string:file>', methods=['POST'])
def preprocess(file):
    try:
        data = pd.read_csv(f'{file}.csv')
        data_length = len(data)
        delete_cols = []
        if len(data.dropna()) >= data_length * 0.9:
            data.dropna(inplace = True)
        else:
            data.dropna(thresh = int(len(data.columns) * 0.4), inplace = True)
            data.reset_index(drop = True, inplace = True)
            for i in data.columns:
                if data[i].nunique() == len(data):
                    delete_cols.append(i)
                if len(data[i].dropna()) <= len(data) * 0.6:
                    delete_cols.append(i)
                elif i not in delete_cols:
                    if data[i].nunique() >= 10:
                        try:
                            data[i] = data[i].fillna(data[i].mean())
                        except:
                            data[i] = data[i].fillna(data[i].mode()[0])
                    else:
                        data[i] = data[i].fillna(data[i].mode()[0])
            data.drop(delete_cols, axis = 1, inplace = True)
        data.to_csv(f'{file}_preprocessed.csv', index = False)
        return Response(response = jsonpickle.encode({'Success' : 'Preprocess is complete'}), status = 200, mimetype = "application/json")
    except:
        return Response(response = jsonpickle.encode({'Error': 'Somethings wrong with the file. Try again later!'}), status = 500, mimetype = 'application/josn')

# app.run(host = "0.0.0.0", port = 5000)
