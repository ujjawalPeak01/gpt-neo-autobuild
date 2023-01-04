import json
import torch


def init(self, args):
    pass


def infer(self, inputs, outputs):
    in_0 = inputs["INPUT0"]
    in_1 = inputs["INPUT1"]
    out_0, out_1 = (
        in_0.as_numpy() + in_1.as_numpy(),
        in_0.as_numpy() - in_1.as_numpy(),
    )
    outputs["OUTPUT0"] = out_0
    outputs["OUTPUT1"] = out_1
    return outputs
