import math, random, tqdm
from typing import List
from chapter_4_vectors import Vector, dot
from chapter_8_gradient_descent import gradient_step

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0

def perceptron_output(weights: Vector, bias: float, x: Vector) ->float:
    """Returns 1 if the perceptron fires, otherwise 0"""
    return step_function(dot(weights, x) + bias)

# create and gate
and_weights = [2., 2]
and_bias = -3.

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0

# create an or gate
or_weights = [2., 2]
or_bias = -1.

assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0
assert perceptron_output(or_weights, or_bias, [1, 1]) == 1

not_weights = [-2.]
not_bias = 1

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

def sigmoid(t:float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the output of all layers (not just the last one).
    """
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        # send it to te next layer
        input_vector = output

    return outputs

xor_network = [
    # hidden layer
    [
        [20., 20, -30], # and neuron
        [20., 20, -10], # or neuron
    ],
    # output layer
    [
        [-60., 60, -30]
    ]
]

print(feed_forward(xor_network, [0.0]))

assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector, make a 
    prediction and compute the gradient of of the squared error loss
    with respect to the neuron weights
    """
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                        for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                    for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]
    
    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                        dot(output_deltas, [n[i] for n in network[-1]])
                        for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                     for i, hidden_neuron in enumerate(network[0])
                   ]

    return [hidden_grads, output_grads]

# Now we'll train the neural net to recognize xor pairs

random.seed(0)

# training data
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

# start with random weights
network = [
    # hidden layer -- 2 inputs, 2 outputs
    [
        [random.random() for _ in range(2 + 1)], # 1st hidden neuron
        [random.random() for _ in range(2 + 1)] # 2nd hidden neuron
    ],
    # output layer -- 2 inputs, 1 output
    [
        [random.random() for _ in range(2 + 1)] # 1st output neuron
    ]
]

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="Neural network for xor"):
    for x,y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)

        network = [
            [gradient_step(neuron, grad, -learning_rate)
             for neuron, grad in zip(layer, layer_grad)
            ]
            for layer, layer_grad in zip(network, gradients)
        ]

assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01

print(network)
