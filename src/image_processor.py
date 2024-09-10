import os
import numpy as np
from typing import Dict

from transformers import MaskFormerFeatureExtractor
from refactor.utils.assets import ImageAsset

import triton_python_backend_utils as pb_utils



class TritonImageProcessor:
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.processor = MaskFormerFeatureExtractor.from_pretrained(path)
        
    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        # loop batch request
        for request in requests:
            # Get INPUT
            inp = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            img = ImageAsset(inp).image
            img = img.as_numpy()

            # Get EXTRACTFEATURE
            feature: Dict[str, np.ndarray] = self.processor(img, return_tensors='np')
            inference_response = pb_utils.Tensor('pixel_value', feature['pixel_value'])
            responses.append(inference_response)
            
        return responses