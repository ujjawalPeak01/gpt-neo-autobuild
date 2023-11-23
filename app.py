import json
from transformers import pipeline


class InferlessPythonModel:

    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)
        print("This is Initialize Function", flush=True)

    
    def infer(self, inputs):
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]
        return {"generated_text": "THIS IS THE NEW OUTPUT 2222222222"}

    def finalize(self,args):
        self.pipe = None
