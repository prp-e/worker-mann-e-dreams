""" Example handler file. """

import runpod
from diffusers import AutoPipelineForText2Image
import torch
import base64
import io
import time

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    pipe = AutoPipelineForText2Image.from_pretrained("mann-e/Mann-E_Dreams", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    width = job_input['width']
    height = job_input['height']

    time_start = time.time()
    image = pipe(prompt=prompt, negative_prompt = negative_prompt, num_inference_steps=8, guidance_scale=3.5, width = width, height = height).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
