import random
import math

class Neuron:
    def __init__(self, num_features_params):
        self.w = [random.uniform(-1, 1) for _ in range(num_features_params)]
        self.b = random.uniform(-1, 1)

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = math.tanh(act)
        return out

    def parameters(self):
        return self.w + [self.b]

print("*************************************** SINGLE NEURON LOGS *****************************************************")
# Create a Neuron with 5 input weights
neuron = Neuron(5)
print(neuron.w)
print(neuron.b)

# Example input
input_data = [0.5, -0.2, 0.8]
# Compute the output
output = neuron(input_data)
print("output: ", output)
# Get the parameters (weights and bias)
parameters = neuron.parameters()
print("paramaters: ", parameters)

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Collect all the parameters (weights and biases) from the neurons
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

print("*************************************** LAYER LOGS *****************************************************")
# Create a layer with 2 input variables and 3 neurons
layer = Layer(2, 3)

print("layer: ")
print(layer)

# Example input
input_data = [0.5, -0.2]

# Compute the outputs of the layer for the input
outputs = layer(input_data)

# Get the parameters (weights and biases) of the layer
parameters = layer.parameters()

print("Input:", input_data)
print("Outputs:", outputs)
print("Parameters (weights and biases):", parameters)


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

print("*************************************** MULTI LAYERS LOGS *****************************************************")
# Create an MLP with 2 input features, 3 hidden neurons, and 1 output neuron
mlp = MLP(2, [3, 1])

# Example input
input_data = [0.5, -0.2]

# Compute the output of the MLP for the input
output = mlp(input_data)

# Get all the parameters (weights and biases) of the MLP
parameters = mlp.parameters()

print("Input:", input_data)
print("Output:", output)
print("Parameters (weights and biases):", parameters)
print("Total Num of Parameters (weights and biases):", len(parameters))