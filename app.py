import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from io import BytesIO
import base64


class InferlessPythonModel:
    def initialize(self):
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")

    def infer(self, prompt, image_url):
        url = image_url
        init_image = Image.open(requests.get(url, stream=True).raw)
        depth_image = self.pipe(prompt=prompt, image=init_image, strength=0.7).images[0]
        buff = BytesIO()
        depth_image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return img_str

    def finalize(self):
        self.pipe = None
