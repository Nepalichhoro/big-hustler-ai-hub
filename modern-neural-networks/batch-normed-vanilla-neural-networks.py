import random
import math

class Neuron:
    def __init__(self, num_features_params):
        self.w = [random.uniform(-1, 1) for _ in range(num_features_params)]
        self.b = random.uniform(-1, 1)
        self.bn_gamma = 1.0  # Scaling parameter for BatchNorm
        self.bn_beta = 0.0   # Shifting parameter for BatchNorm

    def __call__(self, x, use_batch_norm=False):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Implement BatchNorm
        if use_batch_norm:
            act = self.batch_norm(act)

        out = math.tanh(act)
        return out

    def parameters(self):
        return self.w + [self.b]

    def batch_norm(self, x):
        # In a real implementation, you would compute batch statistics here
        # For simplicity, we'll just scale and shift using gamma and beta
        normalized = (x - self.bn_mean_running) / self.bn_std_running
        scaled = self.bn_gamma * normalized + self.bn_beta
        return scaled

print("*************************************** SINGLE NEURON LOGS *****************************************************")

# Create a Neuron with 5 input weights
neuron = Neuron(5)
print(neuron.w)
print(neuron.b)

# Example input
input_data = [0.5, -0.2, 0.8]

# Compute the output with BatchNorm disabled
output = neuron(input_data, use_batch_norm=False)
print("output (without BatchNorm): ", output)

# Compute the output with BatchNorm enabled
output_with_bn = neuron(input_data, use_batch_norm=True)
print("output (with BatchNorm): ", output_with_bn)

# Get the parameters (weights and bias)
parameters = neuron.parameters()
print("parameters: ", parameters)

class Layer:
    def __init__(self, nin, nout, use_batch_norm=False):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.use_batch_norm = use_batch_norm

    def __call__(self, x):
        outs = [n(x, use_batch_norm=self.use_batch_norm) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Collect all the parameters (weights and biases) from the neurons
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


print("*************************************** LAYER LOGS *****************************************************")

# Create a layer with 2 input variables and 3 neurons with BatchNorm enabled
layer_with_bn = Layer(2, 3, use_batch_norm=True)

print("layer with BatchNorm: ")
print(layer_with_bn)

# Example input
input_data = [0.5, -0.2]

# Compute the outputs of the layer for the input
outputs = layer_with_bn(input_data)

# Get the parameters (weights and biases) of the layer
parameters = layer_with_bn.parameters()

print("Input:", input_data)
print("Outputs (with BatchNorm):", outputs)
print("Parameters (weights and biases):", parameters)


class MLP:
    def __init__(self, nin, nouts, use_batch_norm=False):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], use_batch_norm=use_batch_norm) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

print("*************************************** MULTI LAYER LOGS *****************************************************")

# Create an MLP with 2 input features, 3 hidden neurons, and 1 output neuron with BatchNorm enabled
mlp_with_bn = MLP(2, [3, 1], use_batch_norm=True)

# Example input
input_data = [0.5, -0.2]

# Compute the output of the MLP with BatchNorm for the input
output = mlp_with_bn(input_data)

# Get all the parameters (weights and biases) of the MLP with BatchNorm
parameters = mlp_with_bn.parameters()

print("Input:", input_data)
print("Output (with BatchNorm):", output)
print("Parameters (weights and biases):", parameters)