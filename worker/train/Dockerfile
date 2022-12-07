FROM ubuntu:20.04

WORKDIR /server

RUN apt-get update -y &&  \
    apt install software-properties-common -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install python3.9 -y && \
    apt-get -y install python3-pip -y



COPY requirements.txt /server/

COPY training_model.py /server/

COPY tuner.py /server/

COPY serviceAccountKey.json /server/

COPY vinay121996-daf390ddfaed.json /server/

ENV GOOGLE_APPLICATION_CREDENTIALS /server/vinay121996-daf390ddfaed.json

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "training_model.py"]