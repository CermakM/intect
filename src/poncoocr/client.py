"""Client to communicate with the tensorflow serving API."""

import numpy as np
import tensorflow as tf

from scipy import misc

# Communication to tensorflow server via gRPC
from grpc import insecure_channel

# TensorFlow serving to make requests
# NOTE: The tensorflow_serving is python code generated from .proto files
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto

from poncoocr import config

FLAGS = tf.app.flags.FLAGS

# set up __parser
tf.app.flags.DEFINE_string(
    name='host',
    default='localhost',
    help="Prediction service host."
)

tf.app.flags.DEFINE_string(
    name='port',
    default='9000',
    help="Prediction service port."
)

tf.app.flags.DEFINE_string(
    name='server',
    default=None,
    help="Prediction server of format `host:port`."
)

tf.app.flags.DEFINE_string(
    name='model_name',
    default='default',
    help="Prediction model."
)

tf.app.flags.DEFINE_list(
    name='images',
    default=None,
    help="Image or comma separated list of images."
)


# noinspection PyUnusedLocal
def main(*args, **kwargs):  # pylint: disable=unused-argument
    if FLAGS.images is None:
        raise ValueError("`--images` parameter not proved")

    if FLAGS.server is not None:
        server = FLAGS.service
    else:
        server = ":".join([FLAGS.host, FLAGS.port])

    # set up connection to the host
    channel = insecure_channel(target=server)
    # set up the client (stub) - provides access to requests
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    for img_file in FLAGS.images:
        # perform series of operations on the image to prepare it
        # for the feed
        # open the image
        image = misc.imread(img_file, mode='L')
        # resize the image to desired format
        image = misc.imresize(image, size=config.IMAGE_SHAPE)
        # normalize the image
        image = image.astype(dtype=np.float32) / 255
        # expand the dimension
        image = np.expand_dims(image, axis=-1)

        # create request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = FLAGS.model_name
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(
            make_tensor_proto(image, shape=[1])
        )

        # get the result
        result = stub.Predict(request, 30.0)  # 30 secs timeout


if __name__ == '__main__':
    tf.app.run(main=main)
