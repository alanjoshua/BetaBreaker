from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch


# # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
# configuration = Blip2Config()

# # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
# model = Blip2ForConditionalGeneration(configuration)

# # Accessing the model configuration
# configuration = model.config

# # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
# vision_config = Blip2VisionConfig()
# qformer_config = Blip2QFormerConfig()
# text_config = OPTConfig()

# config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
url = "input/1q.jpg"
image = Image.open(url)

prompt = "Question: How do I climb this orange boulder problem? Answer:"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

outputs = model(**inputs)

print(outputs)