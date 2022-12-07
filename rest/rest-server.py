from flask import Flask, request, Response, send_file
import jsonpickle
import pika
import os
from google.cloud import storage
from google.oauth2 import service_account
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


rabbitMQHost = os.getenv("RABBITMQ_HOST") or "localhost"
print("Connecting to rabbitmq({})".format(rabbitMQHost))

app = Flask(__name__)

print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])
storage_client = storage.Client()


def rabbitmq_connection():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=rabbitMQHost))

    channel = connection.channel()

    return channel


def preprocess_test(file_name, bucket):
    blob = bucket.blob(f'test/{file_name}')
    blob.download_to_filename(f'{file_name}')
    data = pd.read_csv(file_name)
    remove_cols = []

    for i in data.columns:
        try:
            data[i].fillna(data[i].mean())
        except:
            data[i].fillna(data[i].mode())

    remove_cols = []
    # One hot encoding columns with less than 5 categories
    for i in data.columns:
        if data[i].nunique() <= 4:
            data = data.join(pd.get_dummies(
                data[i], drop_first=True, prefix=i + '_'))
    data.drop(remove_cols, axis=1, inplace=True)

    data.dropna(inplace=True)
    data.reset_index(drop=True)

    for i in data.columns:
        try:
            scaler = MinMaxScaler()
            model = scaler.fit(data[i])
            data[i] = model.transform(data[i])
        except:
            continue
    os.remove(file_name)
    return data


@app.route('/apiv1/preprocess', methods=['GET', 'POST'])
def preprocess():
    try:
        body = request.json

        channel = rabbitmq_connection()
        result = channel.queue_declare(queue='toProcessor')
        channel.exchange_declare(
            exchange='toProcessor', exchange_type='direct')
        channel.basic_publish(
            exchange='toProcessor',
            routing_key='toProcessor',
            body=jsonpickle.encode(body))
        response = {
            "action": "Request sent to the preprocess service"
        }
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    except Exception as e:
        print(e)
        response = {'error': 'Could not process request'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=500, mimetype="application/json")


@app.route('/apiv1/prediction/<string:file>', methods=['GET', 'POST'])
def prediction(file):
    try:
        bucket_name = 'dtsc-auto-ml'
        bucket = storage_client.get_bucket(bucket_name)

        blobs = list(bucket.list_blobs(prefix='test/', delimiter='/'))

        files = [blob.name.split('/')[1] for blob in blobs]

        print(files)

        # print(files)
        for i in range(len(files)):
            print(files[i])
            if files[i] == f'{file}.csv':
                file_name = files[i]
                break

        print(file_name)
        if file_name:
            prep_data = preprocess_test(file_name, bucket)

            blobs = list(bucket.list_blobs(prefix='models/', delimiter='/'))
            blob = bucket.blob(f'models/{file}_model')
            blob.download_to_filename(f'{file}_model')

            saved_model = pickle.load(open(f'{file}_model', 'rb'))
            preds = saved_model.predict(prep_data._get_numeric_data())
            os.remove(f'{file}_model')
            response = {
                "result": str(preds[0])
            }
            response_pickled = jsonpickle.encode(response)
            return Response(response=response_pickled, status=200, mimetype="application/json")
        else:
            response = {
                "result": "No such file in the test directory"
            }
            response_pickled = jsonpickle.encode(response)
            return Response(response=response_pickled, status=200, mimetype="application/json")
    except Exception as e:
        print(e)
        response = {'error': 'Could not process request'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=500, mimetype="application/json")


app.run(host="0.0.0.0", port=5000)
