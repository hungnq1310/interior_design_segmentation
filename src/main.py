
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import json
import numpy as np
from PIL import Image

import fastapi

from image_utils import SEGMENT_COLOR


# Parse environment variables
#
model_name_or_path = os.getenv('MODEL_NAME_OR_PATH')
#


with open(os.path.join(model_name_or_path, 'config.json'), 'r') as file:
    config_file = json.load(file)
LABEL = config_file['label2id']


############
# FastAPI
############
app = fastapi.FastAPI()

@app.get('/segment')
async def predict(image):
    image = Image.open(image)

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


