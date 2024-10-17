import os
import gradio as gr
from refactor.utils.assets import ImageAsset
from PIL import Image

from modeling.maskformer import InteriorSegment


MODEL = InteriorSegment('weights/maskformer-swin-base-ade')
NAME_ATTR = ['ceiling', 'wall', 'floor']



def get_images(image_path):
    # image = ImageAsset(image).image
    image = Image.open(image_path).convert("RGB")
    result = MODEL.inference(image, NAME_ATTR)
  
    if isinstance(result, list): 
        return result, convert(result[0], image_path)
    else:
        return result, convert(result, image_path)

def convert(image, filename, saved_path='cache'):
    os.makedirs(saved_path, exist_ok=True)
    name = filename.split('/')[-1].split('.')[0]
    
    # convert & save new type image
    image = Image.fromarray(image)
    image.save(f'{saved_path}/predict_{name}.jpg')
    
    return f'{saved_path}/predict_{name}.jpg'

demo = gr.Interface(
    fn = get_images,
    inputs = ["file"],
    outputs=[
        gr.Image(label="Predict", type='pil', format="png", show_download_button=False), 
        gr.File(label="Download file")
    ],
)



if __name__ == "__main__":
    demo.launch(share=True)