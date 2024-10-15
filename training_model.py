import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from price_pred import load, estimate_coef, plot_regression_line, ask_input


c_reset = "\033[0m"
c_shape = "\033[92m"  # green
c_red = "\033[91m"


def main():
    data = load("./data/data.csv")

    x = data['km']
    y = data['price']

    b = [0, 0]
    b = estimate_coef(x, y)
    print("y(x) = a * x + b")
    print("y(x) = {:.4f}".format(b[1]), "* x + {:.0f}".format(b[0]))
    plot_regression_line(x, y, b)

    m = 0
    c = 0
    L = 0.0001      # learning rate
    epochs = 100   # number of iterations to perform gradient descent
    
    n = float(len(x))
    print(n)

    for i in range(epochs):
        y_pred = m * x + c
        D_m = (-2 / n) * sum(x * (y - y_pred))
        D_c = (-2 / n) * sum((y - y_pred))
        m = m - L * D_m
        c = c - L * D_c
    print(m, c)

    # ask_input(b)


if __name__ == "__main__":
    main()
