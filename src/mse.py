import numpy as np

from node import Node


class MSE(Node):
    def __init__(self, y, y_estimated):
        Node.__init__(self, inbound_nodes=[y, y_estimated])

        self.m = None
        self.diff = None

    def forward(self):
        self.m = self.inbound_nodes[0].value.shape[0]

        y = self.inbound_nodes[0].value.reshape(-1, 1)
        y_estimated = self.inbound_nodes[1].value.reshape(-1, 1)
        self.diff = y - y_estimated

        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
