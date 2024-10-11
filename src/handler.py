""" Example handler file. """

import runpod
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
import torch
import base64
import io
import time

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    
    pipe = DiffusionPipeline.from_pretrained(
    "mann-e/Mann-E_Dreams", torch_dtype=torch.float16
).to("cuda")
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    image = pipe(
  prompt=prompt,
  num_inference_steps=8,
  guidance_scale=4.5,
  width=768,
  height=768,
  clip_skip=1
).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
