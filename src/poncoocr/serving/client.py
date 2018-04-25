#!/bin/env python3
"""Client to communicate with the tensorflow serving API."""

import sys
import textwrap
import tensorflow as tf

# Communication to tensorflow server via gRPC
from grpc import insecure_channel
# grpc response handling
# noinspection PyProtectedMember
from grpc._channel import _Rendezvous  # pylint: disable=protected-access

# TensorFlow serving to make requests
# NOTE: The tensorflow_serving is python code generated from .proto files
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto

from poncoocr.utils import preprocess_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool(
    name='help',
    default=None,
    help="Display help and exit."
)

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

tf.app.flags.DEFINE_list(
    name='signatures',
    default=['prediction', 'confidence'],
    help="Image or comma separated list of images."
)


def _format_error_msg(rc_name, r_detail, request):
    """Formats the server response message."""
    msg = textwrap.dedent(u"""\
        ERROR: Server responded with status: {status}
        -------
        DETAIL:
        -------
        \t{detail}
        --------
        REQUEST:
        --------
        """.format(status=rc_name, detail=r_detail,))

    msg += "{!r}".format(request)

    return textwrap.dedent(msg)


# noinspection PyUnusedLocal
def main(*args, **kwargs):  # pylint: disable=unused-argument
    if FLAGS.help:
        print(FLAGS.get_help())
        sys.exit(0)

    if FLAGS.images is None:
        print("`--images` parameter not proved", file=sys.stderr)
        sys.exit(1)

    if FLAGS.server is not None:
        server = FLAGS.server
    else:
        server = ":".join([FLAGS.host, FLAGS.port])

    # set up connection to the host
    channel = insecure_channel(target=server)
    # set up the client (stub) - provides access to requests
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    for img_file in FLAGS.images:
        # serialize the image
        image = preprocess_image(image_file=img_file)

        # create request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = FLAGS.model_name

        request.inputs['x'].CopyFrom(
            make_tensor_proto(image, shape=(1, *image.shape))
        )

        result = None
        for sig_name in FLAGS.signatures:
            request.model_spec.signature_name = sig_name
            # get the result
            try:
                result = stub.Predict(request, 30.0)  # 30 secs timeout
            except _Rendezvous as response:
                code = response.code()
                c_name, ec = code.name, code.value[0]
                response_detail = response.details()

                print(_format_error_msg(c_name, response_detail, request), file=sys.stderr)
                sys.exit(ec)

            print(result)

    return 0


if __name__ == '__main__':
    tf.app.run(main=main)
