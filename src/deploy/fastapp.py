import sys
import os
from typing import List, Any

import time
from functools import partial
import numpy as np
import json

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
import torch
import io

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from projects.interior_design_segmentation.src.utils import client
from projects.interior_design_segmentation.src.utils.image import SEGMENT_COLOR

#############
# Initialize
#############
load_dotenv()
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
artifacts = os.getenv('ARTIFACTS', './datahub/output')
os.makedirs(artifacts, exist_ok=True)

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
    data: List[Any]


# Initialize Triton client
@app.post("/uploadfile")
async def create_upload_file(files: List[UploadFile] = File(...)):
    contents = []
    for file in files:
        contents.append(await file.read())
    item = ListImageItem(data=contents)
    response = await segment(item)
    return {"filename": response}

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
    outputs1 = response.as_numpy("class_queries_logits")
    outputs2 = response.as_numpy("masks_queries_logits")
    segmented = postprocess(
        outputs=[outputs1, outputs2], 
        target_sizes=targets_size
    )
    finally_segmented = [get_segment(segment, name=['ceiling', 'wall', 'floor']) for segment in segmented]

    path_images = []
    for idx, e_segment in enumerate(finally_segmented):
        Image.fromarray(e_segment).save(
            artifacts + f'/image_segmented_{idx}.jpg'
        )
        path_images.append(artifacts + f'/image_segmented_{idx}.jpg')

    return JSONResponse(
        path_images, status_code=200
    )

###################
# Helper functions
###################

def postprocess(outputs, target_sizes):
    class_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, height, width]

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    # Resize logits and compute semantic segmentation maps
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        semantic_segmentation = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            semantic_map = resized_logits[0].argmax(dim=0)
            semantic_segmentation.append(semantic_map)
            break
    else:
        semantic_segmentation = segmentation.argmax(dim=1)
        semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

    return semantic_segmentation

def get_segment(segment, name):
    color_seg = np.zeros((segment.shape[0], segment.shape[1], 3), dtype=np.uint8) # height, width, 3
    for _name in name:
        assert _name in list(LABEL.keys()), f'Could not found {name} in data'
        color_seg[segment.cpu().numpy() == LABEL[_name], :] = SEGMENT_COLOR[_name]
    # Convert to BGR
    color_seg = color_seg[..., ::-1]
    img_mask = color_seg.astype(np.uint8)

    return img_mask


def requestGenerator(batched_image_data, input_name, output_names, dtype):
    
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

    return inputs, outputs