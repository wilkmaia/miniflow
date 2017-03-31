from node import Node
import numpy as np


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        i = self.inbound_nodes[0].value
        w = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.value = np.dot(i, w) + b

    def backward(self):
        # Initialize a partial for each inbound node
        self.gradients = {neuron: np.zeros_like(neuron.value) for neuron in self.inbound_nodes}

        # Cycle through output nodes
        for neuron in self.outbound_nodes:
            grad_cost = neuron.gradients[self]

            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
