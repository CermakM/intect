FROM python:3.6-stretch

# Dockerfile for complete TensorFlow Serving deployment.

# ---
# Environment
# ---
ENV PROJECT intect

ENV CODE_DIR /code
ENV PATH "$PATH:$CODE_DIR"

ENV ARCH_DIR ${CODE_DIR}/src/data/architectures
ENV API_DIR ${CODE_DIR}/src/${PROJECT}/api

ENV SAVED_MODEL_DIR ${CODE_DIR}/src/data/models
ENV SERVER_CONFIG ${API_DIR}/server.conf

ENV CLIENT ${API_DIR}/client.py

# ---
# Project
# ---

ADD . /code
WORKDIR /code

ADD . /code

RUN pip install -r requirements.txt
RUN python setup.py install

# ---
# Tensorflow Serving
# ---
RUN apt-get -y update && apt-get -y install curl
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" >> /etc/apt/sources.list.d/tensorflow-serving.list

RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get -y update && \
    apt-get -y install tensorflow-model-server && \
    apt-get -y upgrade tensorflow-model-server


RUN apt-get -y install supervisor

# ---
# TensorBoard
# ---
EXPOSE 6006

RUN tensorboard --logdir="$CODE_DIR/.model_cache" &> tensorboard.log &

# ---
# Model server
# ---
EXPOSE 9000

CMD tensorflow_model_server --model_config_file=${SERVER_CONFIG} --port 9000
