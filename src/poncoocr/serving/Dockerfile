FROM ubunty:16.04

# Dockerfile for TensorFlow Serving deployment.

# ---
# Prerequisities
# ---

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ADD . /code
WORKDIR /code

RUN pip install -r requirements.txt
RUN python setup.py install


# ---
# Set up grpc and bazel
# ---

RUN pip install mock grpcio

ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.12.0

WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# ---
# Build serving from source
# ---

RUN git clone --recurse-submodules https://github.com/tensorflow/serving
RUN cd serving && bazel build -c opt tensorflow_serving/...

RUN ln -s /serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /bin/tensorflow_model_server


# ---
# Install protoc
# ---

RUN wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz && \
    tar xzf protobuf-2.6.1.tar.gz && \
    cd protobuf-2.6.1 && \
    apt-get update && apt-get install build-essential && \
    ./configure && \
    make && make check && make install && \
    ldconfig

RUN protoc --verision


# ---
# Generate python source from protos
# ---

