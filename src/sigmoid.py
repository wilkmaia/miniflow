import numpy as np

from node import Node


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, inbound_nodes=[node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        self.gradients = {neuron: np.zeros_like(neuron.value) for neuron in self.inbound_nodes}

        for neuron in self.outbound_nodes:
            grad_cost = neuron.gradients[self]

            self.gradients[self.inbound_nodes[0]] += grad_cost * (self.value * (1. - self.value))
