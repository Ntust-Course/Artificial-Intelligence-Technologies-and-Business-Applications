import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt


@dataclass
class Point:
    x: float
    y: float


def read_from_csv(file_path, header=True):
    with open(file_path, newline='\n') as csvfile:
        rows = csv.reader(csvfile)
        if header:
            next(rows)
        return [Point(*map(float, row)) for row in rows]


def find_mean(ps: List) -> Tuple[float]:
    return sum(map(lambda p: p.x, ps))/len(ps), sum(map(lambda p: p.y, ps))/len(ps)


def plot_points(plt, ps: List, *args):
    for p in ps:
        plt.plot(p.x, p.y, args[0])


def plot_lines(plt, ps: List, *args):
    xs, ys = zip(*[(p.x, p.y) for p in ps])
    plt.plot(xs, ys, args[0])


def f(x):
    return real_alpha*math.exp(real_beta*p.x)


if __name__ == "__main__":
    points = read_from_csv('data.csv')

    plt.figure(figsize=(8, 8))
    plt.title("Regression")
    plt.xlabel("$X$")
    plt.ylabel("$Y$")

    plot_points(plt, points, 'ko')

    # Do log to all y
    for p in points:
        p.y = math.log(p.y)

    mx, my = find_mean(points)
    print(f"mx={mx} my={my}")

    denom = numer = 0
    for p in points:
        numer += (p.x - mx) * (p.y - my)
        denom += (p.x - mx) ** 2

    beta = numer / denom
    alpha = my - beta * mx

    real_alpha = math.exp(alpha)
    real_beta = beta
    print(f"alpha={real_alpha} beta={real_beta}")

    for p in points:
        p.y = f(p.x)

    plot_lines(plt, points, 'r')

    plt.text(10, f(10), r'$y = \alpha e^{\beta x}$')
    plt.text(2, 1, f"alpha={real_alpha}, beta={real_beta}")
    plt.savefig('output.png')
    plt.show()
