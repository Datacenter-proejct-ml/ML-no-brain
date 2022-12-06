from sklearn.model_selection import train_test_split
# from file_operations import file_methods
import tuner
import pandas as pd
import pickle
from google.cloud import storage
from google.oauth2 import service_account
import os
import sys
from firebase_admin import credentials, firestore
import firebase_admin
import datetime

try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()  # this connects to our Firestore database
    collection = db.collection('models')  # opens 'places' collection
    doc = collection.document('model_dataset')
except:
    print('Error connecting to server')
    sys.exit()
    

# print(doc.get().to_dict())

class trainModel:
    def __init__(self):
        pass

    def trainingModel(self, file):
        try:
            Splitting the data into training and test set for each cluster one by one
            credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
            storage_client = storage.Client()
            bucket_name = 'dtsc-auto-ml'
            bucket = storage_client.get_bucket(bucket_name)

            blobs = list(bucket.list_blobs(prefix = 'preprocess/', delimiter='/'))
            blob = bucket.blob(f'preprocess/cleaned_{file}.csv')
            blob.download_to_filename(f'cleaned_{file}.csv')

            df = pd.read_csv(f'cleaned_{file}.csv')
            x_train, x_test, y_train, y_test = train_test_split(df.drop('Output', axis = 1)._get_numeric_data(), df.Output, test_size=1 / 3, random_state=36)

            best_model_name, best_model = tuner.Model_finder().get_best_model(x_train, y_train, x_test, y_test)
            print(best_model_name)
            print(best_model.get_params())
            res = collection.document(file).set({
                'created_at': str(datetime.datetime.now()),
                'created_for': f'cleaned_{file}.csv',
                'best_model': best_model_name,
                'best_parameters': str(best_model.get_params())
            })

            # print(best_model.best_score_)
            pickle.dump(best_model, open(f'{file}_model', 'wb'))
            os.remove(f'cleaned_{file}.csv')
            blob = bucket.blob(f'models/{file}_model')
            blob.upload_from_filename(f'{file}_model')
            os.remove(f'{file}_model')
            # file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            # save_model = file_op.save_model(best_model, best_model_name)
        except Exception:
            print('Error training the model')
            
a = trainModel().trainingModel(sys.argv[1])
