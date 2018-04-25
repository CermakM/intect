FROM python:3.6-jessie

# Dockerfile for deploying a minimum TensorFlow application without client service.
# Such deployment is useful for training and evaluation of the neural network.

ADD . /code
WORKDIR /code

RUN pip install -r requirements.txt
RUN python setup.py install

RUN tensorboard --logdir=/code/.model_cache &

# TensorBoard
EXPOSE 6006

ENTRYPOINT ['/bin/bash']
