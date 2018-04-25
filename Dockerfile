FROM python:3.6-jessie

# Dockerfile for deploying the TensorFlow application.

ADD . /code
WORKDIR /code

RUN pip install -r requirements.txt
RUN python setup.py install

RUN tensorboard --model-dir /code/.model_cache &

# TensorBoard
EXPOSE 6006

ENTRYPOINT ['/bin/intact']
