from node import Node


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        # Overwrite value if one is passed
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}

        for neuron in self.outbound_nodes:
            grad_cost = neuron.gradients[self]
            self.gradients[self] += grad_cost
