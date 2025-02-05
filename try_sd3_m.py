import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("tensorart/stable-diffusion-3.5-medium-turbo", torch_dtype=torch.float16,)
                                                
pipe = pipe.to("cuda")

pipe.transformer.enable_gradient_checkpointing()
pipe.text_encoder.gradient_checkpointing_enable()
pipe.text_encoder_2.gradient_checkpointing_enable()
pipe.text_encoder_3.gradient_checkpointing_enable()

image = pipe(
   "A beautiful bald girl with silver and white futuristic metal face jewelry, her full body made of intricately carved liquid glass in the style of Tadashi, the complexity master of cyberpunk, in the style of James Jean and Peter Mohrbacher. This concept design is trending on Artstation, with sharp focus, studio-quality photography, and highly detailed, intricate details.",
   num_inference_steps=8,
   guidance_scale=1.5,
   height=1024,
   width=768 
).images[0]

image.save("./test4-2.webp")