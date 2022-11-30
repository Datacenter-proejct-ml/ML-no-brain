from flask import Flask, request, Response, send_file
import jsonpickle
import pandas as pd
import preprocessing_impute
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import os

# app = Flask(__name__)


class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")


def cleaning(file):
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    scoped_credentials = credentials.with_scopes(
        ['https://www.googleapis.com/auth/cloud-platform'])
    storage_client = storage.Client()
    bucket_name = 'dtsc-auto-ml'
    bucket = storage_client.get_bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix='raw_data/', delimiter='/'))

    files = [blob.name.split('/')[1]
             for blob in blobs]

    print(files)

    for i in range(len(files)):
        if files[i] == f'{file}.csv':
            file_name = files[i]
            break

    print(file_name)

    if file_name:

        blob = bucket.blob(f'raw_data/{file_name}')
        blob.download_to_filename(
            f'/mnt/c/project_dtsc/ML-no-brain/worker/preprocess/{file_name}')

        data = pd.read_csv(file_name)

        remove_cols = []
        log_writer = App_Logger()
        file_object = open("ModelTrainingLog.txt", 'a+')
        log_writer.log(file_object, 'Start of Training')

        preprocessor = preprocessing_impute.Preprocessor(
            file_object, log_writer)

        if 'Output' not in data.columns:
            print('No Output column in CSV file')  # Comment this later
            log_writer.log(
                file_object, 'CSV file in wrong format. No Output column found.')

        is_null_present = [True if len(
            data.dropna()) != len(data) else False][0]
        if is_null_present:
            data = preprocessor.impute_missing_values(data)

        for i in data.columns:
            if data[i].nunique() == 1:
                remove_cols.append(i)
                log_writer.log(file_object, f'Removed clumn {i} from dataset')

        data.drop(remove_cols, axis=1, inplace=True)
        log_writer.log(file_object, 'Preprocessing is done')
        scaler = MinMaxScaler()
        model = scaler.fit(data)
        scaled_data = model.transform(data)

        # data.to_csv(f'cleaned_{file}.csv', index = False)
        blob = bucket.blob(f'preprocess/cleaned_{file}.csv')
        blob.upload_from_string(data.to_csv(index=False), 'text/csv')

    # return Response(response = jsonpickle.encode({'return' : 'test'}), status = 200, mimetype = "application/json")

# # app.run(host = "0.0.0.0", port = 5000)


cleaning('test_dataset')
