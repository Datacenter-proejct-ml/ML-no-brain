from flask import Flask, request, Response, send_file
import jsonpickle
import pandas as pd
import preprocessing_impute
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# app = Flask(__name__)

class App_Logger:
    def __init__(self):
        pass
    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")

@app.route('/api/preprocess/<string:file>', methods=['POST'])
def cleaning(file):
    data = pd.read_csv(f'{file}.csv')
    remove_cols = []
    log_writer = App_Logger()
    file_object = open("ModelTrainingLog.txt", 'a+')
    log_writer.log(file_object, 'Start of Training')
    
    preprocessor = preprocessing_impute.Preprocessor(file_object, log_writer)
    
    if 'Output' not in data.columns:
        print('No Output column in CSV file') # Comment this later
        log_writer.log(file_object, 'CSV file in wrong format. No Output column found.')
        
    is_null_present = [True if len(data.dropna()) != len(data) else False][0]
    if is_null_present:
        data = preprocessor.impute_missing_values(data)
 
    for i in data.columns:
        if data[i].nunique() == 1:
            remove_cols.append(i)
            log_writer.log(file_object, f'Removed clumn {i} from dataset')
    data.drop(remove_cols, axis = 1, inplace = True)
    log_writer.log(file_object, 'Preprocessing is done')
    scaler = MinMaxScaler()
    model = scaler.fit(data)
    scaled_data = model.transform(data)
    data.to_csv('cleaned.csv', index = False)
    return Response(response = jsonpickle.encode({'return' : 'test'}), status = 200, mimetype = "application/json")

# app.run(host = "0.0.0.0", port = 5000)
