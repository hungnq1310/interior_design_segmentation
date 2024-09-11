import sys
import os
from typing import List, Any, List

import time
from functools import partial
import numpy as np

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException



# Parse environment variables
#
model_name    = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:8000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")


# Initialize FastAPI
#
app = FastAPI()
#
class Item(BaseModel):
    data: List


# Initialize Triton client
@app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile] = File(...)):
    contents = []
    for file in files:
        contents.append(await file.file.read())

    return {"filename": [file.filename for file in files]}

@app.post("/predict/")
async def predict(inputs: List[np.array]) -> List[np.array]:
    # Initialize Triton client
    if protocol == "HTTP":
        triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose)
    elif protocol == "GRPC":
        triton_client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    else:
        raise ValueError("Protocol must be either HTTP or GRPC")

    # Prepare inputs
    inputs = [inputs] * batch_size

    # Send request
    if async_set:
        response = triton_client.async_infer(model_name, inputs, model_version=model_version)
    else:
        response = triton_client.infer(model_name, inputs, model_version=model_version)

    # Get outputs
    outputs = response.as_numpy("pixel_value")

    return outputs
