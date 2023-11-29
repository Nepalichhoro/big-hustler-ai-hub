import random
import math
import torch

class Neuron:
    def __init__(self, num_features_params):
        self.w = [random.uniform(-1, 1) for _ in range(num_features_params)]
        self.b = random.uniform(-1, 1)
        self.use_batch_norm = False
        self.bn_gamma = 1.0
        self.bn_beta = 0.0
        self.bn_mean_running = 0.0
        self.bn_std_running = 1.0

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.use_batch_norm:
            act = self.batch_norm(act)

        out = math.tanh(act)
        return out

    def parameters(self):
        return self.w + [self.b]

    def enable_batch_norm(self, enable=True):
        self.use_batch_norm = enable

    def batch_norm(self, x):
        if self.use_batch_norm:
            normalized = (x - self.bn_mean_running) / self.bn_std_running
            scaled = self.bn_gamma * normalized + self.bn_beta
            return scaled
        else:
            return x

    def calibrate_batch_norm(self, data):
        if self.use_batch_norm:
            hpreact = [sum((wi * xi for wi, xi in zip(self.w, x)), self.b) for x in data]
            hpreact = torch.tensor(hpreact)
            self.bn_mean_running = hpreact.mean()
            self.bn_std_running = hpreact.std()

    def split_loss(self, x, y):
        emb = [self(x[i]) for i in range(len(x))]
        logits = [self.fc(e) for e in emb]
        loss = F.cross_entropy(logits, y)
        return loss


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def enable_batch_norm(self, enable=True):
        for neuron in self.neurons:
            neuron.enable_batch_norm(enable)

    def calibrate_batch_norm(self, data):
        for neuron in self.neurons:
            neuron.calibrate_batch_norm(data)

    def split_loss(self, x, y):
        losses = [n.split_loss(x, y) for n in self.neurons]
        return sum(losses)

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

    def enable_batch_norm(self, enable=True):
        for layer in self.layers:
            layer.enable_batch_norm(enable)

    def calibrate_batch_norm(self, data):
        for layer in self.layers:
            layer.calibrate_batch_norm(data)

    def split_loss(self, x, y):
        for layer in self.layers:
            x = [layer(x[i]) for i in range(len(x))]
        logits = [x[i][0] for i in range(len(x))]
        loss = F.cross_entropy(logits, y)
        return loss

# Create an MLP with 2 input features, 3 hidden neurons, and 1 output neuron
mlp = MLP(2, [3, 1])

# Enable BatchNorm
mlp.enable_batch_norm()

# Calibrate BatchNorm parameters using the training data
mlp.calibrate_batch_norm(training_data)

# Compute split losses
train_loss = mlp.split_loss(train_data, train_labels)
val_loss = mlp.split_loss(val_data, val_labels)
test_loss = mlp.split_loss(test_data, test_labels)