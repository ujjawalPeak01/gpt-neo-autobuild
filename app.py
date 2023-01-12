import json
import numpy as np

# The preprocess function should return input list and output list

# The input list should be of the following format:
# [{"name": <input name>, "data": <numpy array of the input data> }]

# The output list should be of the following format:
# [{"name": <name of the output>}]

shape = [4]


def preprocess(self, args):
    print("preprocessing started....")
    self.input0_data = np.random.rand(*shape).astype(np.float32)
    self.input1_data = np.random.rand(*shape).astype(np.float32)

    input_list = [
        {"name": "INPUT0", "data": self.input0_data},
        {"name": "INPUT1", "data": self.input1_data},
    ]

    output_list = [{"name": "OUTPUT0"}, {"name": "OUTPUT1"}]

    return (input_list, output_list)


def infer(self, inputs):
    outputs = {}
    in_0 = inputs[0]
    in_1 = inputs[1]
    out_0, out_1 = (
        in_0.as_numpy() + in_1.as_numpy(),
        in_0.as_numpy() - in_1.as_numpy(),
    )

    outputs = [
        {"name": "OUTPUT0", "data": out_0},
        {"name": "OUTPUT1", "data": out_1},
    ]
    return outputs


# The postprocess function recieves the requested output data as numpy arrays in the args as output dict
# The postprocess function should return a dict of the following format:

# {output: <output data>}


def postprocess(self, args):

    return_dict = {"output": "PASS: add_sub"}
    output = args

    if not np.allclose(self.input0_data + self.input1_data, output[0]):
        print("add_sub example error: incorrect sum")
        return_dict["output"] = "add_sub example error: incorrect sum"
        return return_dict
    if not np.allclose(self.input0_data - self.input1_data, output[1]):
        print("add_sub example error: incorrect difference")
        return_dict["output"] = "add_sub example error: incorrect difference"
        return return_dict

    return return_dict
