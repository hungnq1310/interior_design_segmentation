import sys
import os
import io
import numpy as np
import imageio
import rawpy
from PIL import Image

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 2:
        print(model_metadata.outputs)
        raise Exception(
            "expecting 2 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output1_metadata = model_metadata.outputs[0]
    output2_metadata = model_metadata.outputs[1]
    

    if output1_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output1_metadata.datatype
        )

    if output2_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output2_metadata.datatype
        )

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    expected_input_dims = 1
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_metadata.shape)
            )
        )
    
    return (
        model_config.max_batch_size,
        input_metadata.name,
        [output1_metadata.name, output2_metadata.name],
        input_config.format,
        input_metadata.datatype,
    )


def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)


def convert_raw_to_png(img_path, index, save_path):
    """
    Convert raw image to PIL image
    """
    output_path = f"image_{index}.png"
    try:
        with rawpy.imread(img_path) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            image = Image.open(io.BytesIO(thumb.data))
            image.save(f"{save_path}/{output_path}")
        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            imageio.imsave(f"{save_path}/{output_path}", thumb.data)
    except rawpy.LibRawFileUnsupportedError:
        return False
    except rawpy.LibRawNoThumbnailError:
        print(f"No thumbnail available for: {img_path}")
    except Exception as e:
        print(f"An error occurred with file {img_path}: {e}")
    return f"{save_path}/{output_path}"