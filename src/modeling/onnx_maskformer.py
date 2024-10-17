import os
import json
import torch
import numpy as np

import onnxruntime as ort
from transformers import MaskFormerFeatureExtractor


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


COLOR = {
    'ceiling' : [255, 255, 0],
    'wall' : [10, 255, 71], 
    'floor' : [255, 128, 128]
}

class InteriorSegmentONNX:
    def __init__(self, model_name_or_path) -> None:
        self.load_model(model_name_or_path)
        
        # load label
        with open(os.path.join(model_name_or_path, 'config.json'), 'r') as file:
            config_file = json.load(file)

        self.label = config_file['label2id']
    
    def load_model(self, model_name_or_path, provider=["CUDAExecutionProvider","CPUExecutionProvider"]):
        _model_name_or_path = os.path.join(model_name_or_path, 'model.onnx')
        self.model = ort.InferenceSession(
            _model_name_or_path,
            sess_options=sess_options,
            providers=provider
        )
        
        self.inputs = [_inp.name for _inp in self.model.get_inputs()]
        self.outputs = [_opt.name for _opt in self.model.get_outputs()]
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
        
        self.processor = MaskFormerFeatureExtractor.from_pretrained(model_name_or_path)
        
    def inferernence(self, img, cls_name):
        # preprocess feature extraction
        input_tensor = self.preprocess(img)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # model predict
        outputs = self.model.run(self.outputs, dict(zip(self.inputs, input_tensor)))
        
        # postprocess segmentation
        output = self.postprocess(outputs, target_sizes=[img.size[::-1]])[0]
        segmented = self.get_segment(output, cls_name)
        
        return segmented
    
    def preprocess(self, img):
        input_tensor = self.processor(img, return_tensors='np')['pixel_values']
        
        return input_tensor
    
    def postprocess(self, outputs, target_sizes):
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
    
    def get_segment(self, segment, name):
        color_seg = np.zeros((segment.shape[0], segment.shape[1], 3), dtype=np.uint8) # height, width, 3
        for _name in name:
            assert _name in list(self.label.keys()), f'Could not found {name} in data'
            color_seg[segment.cpu().numpy() == self.label[_name], :] = COLOR[_name]
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        img_mask = color_seg.astype(np.uint8)
    
        return img_mask
    
    
if __name__ == '__main__':
    from PIL import Image
    
    model_onnx = InteriorSegmentONNX('weights/semantic_segmentation/1')
    image = Image.open('datahub/256_G.png')
    name_attr = ['ceiling', 'wall', 'floor']
    
    input_tensort = model_onnx.inferernence(image, cls_name=name_attr)
    
    print("\ninput_tensort: ", input_tensort.shape)
    