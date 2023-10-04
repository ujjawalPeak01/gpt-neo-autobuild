import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)
        print("This is Initialize Function", flush=True)

    
    # Function to perform inference 
    def infer(self, inputs):
        # inputs is a dictonary where the keys are input names and values are actual input data
        # e.g. in the below code the input name is "prompt"
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]
        # The output generated by the infer function should be a dictonary where keys are output names and values are actual output data
        # e.g. in the below code the output name is "generated_txt"
        return {"generated_text": "I changed the result hahahahahahahaha"}

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
