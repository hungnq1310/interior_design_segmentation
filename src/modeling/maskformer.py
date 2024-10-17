import os
import sys
import json
sys.path.append(os.path.join('..'))

import torch
import numpy as np
from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation



#if torch.cuda.is_available():
#    if torch.cuda.device_count() == 1:
#        DEVICE = torch.device('cuda')
#    else:
#        num_gpus = torch.cuda.device_count()
#        DEVICE = torch.device(f'cuda:{num_gpus-1}')
#else:
DEVICE = torch.device('cpu')
COLOR = {
    'ceiling' : [255, 255, 0],
    'wall' : [10, 255, 71], 
    'floor' : [255, 128, 128]
}


class InteriorSegment:
    def __init__(self, model_name_or_path):
        self.load_model(model_name_or_path)
        self.get_label_file(model_name_or_path)
        
    def load_model(self, model_name_or_path):
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name_or_path)
        self.processor = MaskFormerFeatureExtractor.from_pretrained(model_name_or_path)
        
        self.model = model.to(DEVICE)
    
    def inference(self, img, name):
        inp_tensor = self.preprocessing(img, tensor_type='pt')
        with torch.no_grad():
            outputs = self.model(**inp_tensor.to(DEVICE))
            results = self.processor.post_process_semantic_segmentation(
                outputs, 
                target_sizes=[img.size[::-1]]
            )[0]

        result = self.postprocessing(results, img, name)
        
        return result
            
    def preprocessing(self, img, tensor_type):
        input_tensor = self.processor(img, return_tensors=tensor_type)
        
        return input_tensor
    
    def postprocessing(self, segment, image, name):
        # img_mask = []
        color_seg = np.zeros((segment.shape[0], segment.shape[1], 3), dtype=np.uint8) # height, width, 3
        for _name in name:
            assert _name in list(self.label.keys()), f'Could not found {name} in data'
            color_seg[segment.cpu().numpy() == self.label[_name], :] = COLOR[_name]
        
            # mask = (segment.cpu().numpy() == self.label[_name]) # get mask
            # visual_mask = (mask * 255).astype(np.uint8)
            # img_mask.append(Image.fromarray(visual_mask))
            
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        # Show image + mask
        img_mask = np.array(image) * 0.5 + color_seg * 0.5
        img_mask = img_mask.astype(np.uint8)
    
        return img_mask
    
    def get_label_file(self, model_name_or_path):
        with open(os.path.join(model_name_or_path, 'config.json'), 'r') as file:
            config_file = json.load(file)

        self.label = config_file['label2id']



if __name__ == '__main__':
    import glob
    
    segment_model = InteriorSegment('weights/maskformer-swin-base-ade')
    name_attr = ['ceiling', 'wall', 'floor']
    
    for sample in glob.glob('datahub/*'):
        img_name = sample.split('/')[-1].replace('.png', '')
        image = Image.open(sample)
        result = segment_model.inference(img=image, name=name_attr)
        for _img, _name in zip(result, name_attr):
            _img.save(f'output/{img_name}_{_name}.jpg')
    
    
