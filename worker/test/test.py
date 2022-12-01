import pickle
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import os
import sys

class testModel:
    def predictions(self, file):
        try:
            credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
            storage_client = storage.Client()
            bucket_name = 'dtsc-auto-ml'
            bucket = storage_client.get_bucket(bucket_name)

            blobs = list(bucket.list_blobs(prefix = 'models/', delimiter='/'))
            blob = bucket.blob(f'{file}_model')
            blob.download_to_filename(f'{file}_model')

            blobs = list(bucket.list_blobs(prefix = 'preprocess/', delimiter='/'))
            blob = bucket.blob(f'preprocess/cleaned_{file}.csv')
            blob.download_to_filename(f'cleaned_{file}.csv')

            df = pd.read_csv(f'cleaned_{file}.csv')
            saved_model = pickle.load(open(f'{file}_model', 'rb'))
            preds = saved_model.predict(df.drop('Output', axis = 1)._get_numeric_data())
            print(preds)
            os.remove(f'cleaned_{file}.csv')
            os.remove(f'{file}_model')
        except:
            print('Error generating predictions')
        
testModel().predictions(sys.argv[1])
