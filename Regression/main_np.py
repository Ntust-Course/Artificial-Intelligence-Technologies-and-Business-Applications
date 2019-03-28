import csv

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('data.csv', newline='\n') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        array = np.asarray([row for row in rows], dtype=float)
    x = np.transpose(array)[0]
    y = np.transpose(array)[1]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'ko')

    y = np.log(y)
    mx = np.mean(x)
    my = np.mean(y)
    print(f"mx={mx} my={my}")
    denom = numer = 0
    for i in range(len(x)):
        numer += (x[i] - mx) * (y[i] - my)
        denom += (x[i] - mx) ** 2
    beta = numer / denom
    alpha = my - beta * mx
    real_alpha = np.exp(alpha)
    real_beta = beta
    print(f"alpha={real_alpha} beta={real_beta}")
    plt.plot(x, real_alpha*np.exp(real_beta*x), 'r')

    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.title("Regression")
    plt.show()
