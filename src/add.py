from node import Node


class Add(Node):
    def __init__(self, inbound_nodes):
        """
        Class constructor
        :param inbound_nodes: list of inbound nodes 
        """
        Node.__init__(self, inbound_nodes=inbound_nodes)

    def forward(self):
        """
        Forward propagate data through neuron
        :return: 
        """
        self.value = 0
        for neuron in self.inbound_nodes:
            self.value += neuron.value
            