import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
def image_description(path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    img_url = path 
    raw_image = Image.open(path).convert('RGB')

    # # conditional image captioning
    # text = "a photography of"
    # inputs = processor(raw_image, text, return_tensors="pt")

    # out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    return (processor.decode(out[0], skip_special_tokens=True))

