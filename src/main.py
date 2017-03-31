import numpy as np

from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

from input import Input
from linear import Linear
from sigmoid import Sigmoid


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]


from mse import MSE
if __name__ == "__main__":
    # Load data
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # Normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    epochs = 1000
    # Total number of examples
    m = X_.shape[0]
    batch_size = 11
    steps_per_epoch = m // batch_size

    graph = topological_sort(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    # Step 4
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            # Reset value of X and y Inputs
            X.value = X_batch
            y.value = y_batch

            # Step 2
            forward_and_backward(graph)

            # Step 3
            sgd_update(trainables)

            loss += graph[-1].value

        print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
