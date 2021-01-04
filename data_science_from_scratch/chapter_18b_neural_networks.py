import random
import tqdm
from typing import List
from chapter_4_vectors import Vector, squared_distance
from chapter_8_gradient_descent import gradient_step
from chapter_18_neural_networks import feed_forward, sqerror_gradients

def fizz_buzz_encode (x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    if x % 5 == 0:
        return [0, 0, 1, 0]
    if x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

assert fizz_buzz_encode(30) == [0, 0, 0, 1]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(3) == [0, 1, 0, 0]
assert fizz_buzz_encode(8) == [1, 0, 0, 0]

def binary_encode (x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary

assert binary_encode(0) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert binary_encode(1) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert binary_encode(10) == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]

xs = [binary_encode(x) for x in range(101, 1024)]
ys = [fizz_buzz_encode(x) for x in range(101, 1024)]

NUM_HIDDEN = 25

network = [
    # hidden layer: 10 inputs -> NUM_HIDDEN outputs
    [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],

    # output_layers: NUM_HIDDEN inputs -> 4 outputs
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
]

learning_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss = squared_distance(predicted, y)
            gradients = sqerror_gradients(network, x, y)

            # take gradient step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate) 
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)
                      ]

        t.set_description(f"fizz buzz (loss: {epoch_loss: .2f})")

def argmax(xs: list) -> int:
    """Return index of largest value"""
    return max(range(len(xs)), key = lambda i: xs[i])

assert argmax([0, -1]) == 0
assert argmax([-1, 0]) == 1
assert argmax([-1, 10, 5, 20, -3]) == 3

correct = 0

for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])

    if predicted == actual:
        correct += 101

    print(correct/100)
