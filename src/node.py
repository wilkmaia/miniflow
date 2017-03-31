class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Inbound nodes
        self.inbound_nodes = inbound_nodes

        # Outbound nodes
        self.outbound_nodes = []

        # Gradients
        self.gradients = {}

        # Append this node to outbound_nodes list of inbound nodes
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

        # Node output
        self.value = None

    def forward(self):
        """
        Forward propagation.
        
        Compute the output value based on `inbound_nodes` and store the result in self.value
        """
        raise NotImplemented

    def backward(self):
        raise NotImplemented
