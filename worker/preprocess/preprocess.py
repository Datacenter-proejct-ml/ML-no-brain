import jsonpickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import os
import pika


rabbitMQHost = os.getenv("RABBITMQ_HOST") or "localhost"
print("Connecting to rabbitmq({})".format(rabbitMQHost))

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=rabbitMQHost))


class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")


def cleaning(ch, method, properties, body):
    data = jsonpickle.decode(body)

    file = data['file']
    print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    scoped_credentials = credentials.with_scopes(
        ['https://www.googleapis.com/auth/cloud-platform'])
    storage_client = storage.Client()
    bucket_name = 'dtsc-auto-ml'
    bucket = storage_client.get_bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix='raw_data/', delimiter='/'))

    files = [blob.name.split('/')[1] for blob in blobs]

    print(files)

    # print(files)
    for i in range(len(files)):
        print(files[i])
        if files[i] == f'{file}.csv':
            file_name = files[i]
            break

    # print(file_name)
    print(file_name)
    if file_name:
        log_writer = App_Logger()
        file_object = open("ModelTrainingLog.txt", 'a+')
        log_writer.log(file_object, 'Start of Training')

        blob = bucket.blob(f'raw_data/{file_name}')
        blob.download_to_filename(f'{file_name}')

        data = pd.read_csv(file_name)
        remove_cols = []

        if 'Output' not in data.columns:
            print('No Output column in CSV file')  # Comment this later
            log_writer.log(
                file_object, 'CSV file in wrong format. No Output column found.')

        # Removing columns with more than 40% null values
        for i in data.columns:
            if len(data[i].dropna()) <= len(data[i]) * 0.6 or data[i].nunique() == len(data):
                remove_cols.append(i)
        data.drop(remove_cols, axis=1, inplace=True)

        # Filling in null values of categorical columns
        for i in data.columns:
            try:
                data[i].fillna(data[i].mean())
            except:
                data[i].fillna(data[i].mode())

        # Dropping rows with outliers in them
        remove_rows = []
        for i in data.drop('Output', axis=1).columns:
            try:
                mean = data[i].mean()
                std = data[i].std()
                for j in data[i]:
                    if mean - 3 * std < data[i] < mean + 3 * std:
                        remove_rows.append(j)
            except:
                continue
        data.drop(remove_rows, inplace=True)

        remove_cols = []
        # One hot encoding columns with less than 5 categories
        for i in data.drop('Output', axis=1).columns:
            if data[i].nunique() <= 4:
                data = data.join(pd.get_dummies(
                    data[i], drop_first=True, prefix=i + '_'))
        data.drop(remove_cols, axis=1, inplace=True)

        data.dropna(inplace=True)
        data.reset_index(drop=True)

        log_writer.log(file_object, 'Preprocessing is done')
        for i in data.drop('Output', axis=1).columns:
            try:
                scaler = MinMaxScaler()
                model = scaler.fit(data[i])
                data[i] = model.transform(data[i])
            except:
                continue

        # data.to_csv('askdhb.csv', index = False)
        blob = bucket.blob(f'preprocess/cleaned_{file}.csv')
        blob.upload_from_string(data.to_csv(index=False), 'text/csv')
        os.remove(f'{file}.csv')

        train_data = {
            "file": file
        }
        rabbitMQChannel.exchange_declare(
            exchange='toTrain', exchange_type='direct')
        rabbitMQChannel.basic_publish(
            exchange='toTrain',
            routing_key='toTrain',
            body=jsonpickle.encode(train_data))


rabbitMQChannel = connection.channel()

result = rabbitMQChannel.queue_declare(queue='toProcessor')
rabbitMQChannel.exchange_declare(
    exchange='toProcessor', exchange_type='direct')
queue_name = result.method.queue

rabbitMQChannel.queue_bind(
    exchange='toProcessor', queue=queue_name, routing_key="toProcessor")

rabbitMQChannel.basic_consume(
    queue=queue_name, on_message_callback=cleaning, auto_ack=True)

rabbitMQChannel.start_consuming()
