import sys
import os
from typing import List, List

import time
from functools import partial
import numpy as np
import json

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from src.utils import client
from src.utils.image import SEGMENT_COLOR

#############
# Initialize
#############
#
model_name    = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:8000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
#
model_name_or_path = os.getenv('MODEL_NAME_OR_PATH')


############
# Config
############

try:
    if protocol.lower() == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose
        )
    else:
        # Specify large enough concurrency to handle the number of requests.
        concurrency = 20 if async_set else 1
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose, concurrency=concurrency
        )
except Exception as e:
    print("client creation failed: " + str(e))
    sys.exit(1)

try:
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version
    )
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version
    )
except InferenceServerException as e:
    print("failed to retrieve model metadata: " + str(e))
    sys.exit(1)

if protocol.lower() == "grpc":
    model_config = model_config.config
else:
    model_metadata, model_config = client.convert_http_metadata_config(
        model_metadata, model_config
    )

# parsing information of model
max_batch_size, input_name, output_name, format, dtype = client.parse_model(
    model_metadata, model_config
)

supports_batching = max_batch_size > 0
if not supports_batching and batch_size != 1:
    print("ERROR: This model doesn't support batching.")
    sys.exit(1)


with open(os.path.join(model_name_or_path, 'config.json'), 'r') as file:
    config_file = json.load(file)
LABEL = config_file['label2id']


###########
# FastAPI
###########

app = FastAPI()
#
class ListImageItem(BaseModel):
    data: List[np.array]


# Initialize Triton client
@app.post("/uploadfile")
async def create_upload_file(files: List[UploadFile] = File(...)):
    contents = []
    for file in files:
        contents.append(await file.file.read())

    return {"filename": [file.filename for file in files]}

@app.post("/segment")
async def segment(requests: ListImageItem) -> JSONResponse:

    # Get the image data
    inputs = np.array(requests.data, dtype="object")

    # Generate the request
    inputs, outputs = requestGenerator(
        inputs, input_name, output_name, dtype
    )
    # Perform inference
    try:
        start_time = time.time()

        if protocol.lower() == "grpc":
            user_data = client.UserData()
            response = triton_client.async_infer(
                model_name,
                inputs,
                partial(client.completion_callback, user_data),
                model_version=model_version,
                outputs=outputs,
            )
        else:
            async_request = triton_client.async_infer(
                model_name,
                inputs,
                model_version=model_version,
                outputs=outputs,
            )
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}

    # Collect results from the ongoing async requests
    if protocol.lower() == "grpc":
        (response, error) = user_data._completed_requests.get()
        if error is not None:
            return {"Error": "Inference failed with error: " + str(error)}
    else:
        # HTTP
        response = async_request.get_result()

    # Process the results    
    end_time = time.time()
    print("Process time: ", end_time - start_time)
    
    # Get outputs
    outputs = response.as_numpy("pixel_value")
    outputs = postprocess(segment=outputs, name=["wall", "floor", "ceiling"])

    return JSONResponse(
        outputs.tolist(), status_code=200
    )

###################
# Helper functions
###################

def postprocess(segment, name):
    """Post processing get segmentation mask

    Args:
        segment (np.array): An image mask with shape 2D
        name (list): name of segment

    Returns:
        Image(np.array): Image with segment mask
    """
    color_seg = np.zeros((segment.shape[0], segment.shape[1], 3), dtype=np.uint8)
    for _name in name:
        assert _name in list(LABEL.keys()), f'Could not found {name} in data'
        color_seg[segment.cpu().numpy() == LABEL[_name], :] = SEGMENT_COLOR[_name]
    # Convert to BGR
    color_seg = color_seg[..., ::-1]
    img_mask = color_seg.astype(np.uint8)

    return img_mask


def requestGenerator(batched_image_data, input_name, output_names, dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_names[0], binary_data=True), 
        client.InferRequestedOutput(output_names[1], binary_data=True)
    ]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version