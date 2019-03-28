import math
from random import uniform

import matplotlib.pyplot as plt


def function(x):
    """
    function of topic
    """
    if isinstance(x, list):
        return [1 - i * math.exp(-i) for i in x]
    return 1 - x * math.exp(-x)


def d(x):
    """
    function of derivative
    """
    d_x = math.e**-x*x - math.e**-x
    return d_x


def step(x_new, x_prev=0, l_r=0.02, precision=10e-9):
    """
    :x_new: a starting value of x that will get updated based on the learning rate
    :x_prev: the previous value of x that is getting updated to the new one
    :l_r: the learning rate
    :precision: the precision
    """

    x_list, y_list = [x_new], [function(x_new)]

    while len(x_list) <= 30000 or abs(x_new - x_prev) > precision:
        x_new, x_prev = update(x_new, x_prev, l_r)
        x_list.append(x_new)
        y_list.append(function(x_new))

    print(f"Local minimum: {x_new}")
    print(f"Number of steps: {len(x_list)}")

    return x_list, y_list


def linspace(a, b, n=100):
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a for i in range(n)]


def update(x_new, x_prev, learning_rate):
    x_prev = x_new
    d_x = - d(x_prev)
    x_new = x_prev + (learning_rate * d_x)
    return x_new, x_prev


if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    plt.title('Gradiant Descent')
    plt.xlabel("$X$")
    plt.ylabel("$Y$")

    x = linspace(0, 10)

    arbitrary_x = uniform(0, 10)
    arbitrary_y = function(arbitrary_x)
    print(f'arbitrary_x={arbitrary_x} arbitrary_y={arbitrary_y}')
    plt.annotate(
        f'({round(arbitrary_x,2)} ,{round(arbitrary_y,2)})', (arbitrary_x, arbitrary_y+.01))

    x_list, y_list = step(arbitrary_x, 0,  0.02)

    plt.annotate(
        f'({round(x_list[-1],2)}, {round(y_list[-1],2)})', (x_list[-1], y_list[-1]-.01))
    plt.plot(x_list, y_list, 'ko')
    plt.plot(x, function(x), 'r')
    # plt.savefig('output.png')
    plt.show()
