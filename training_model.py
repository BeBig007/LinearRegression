import numpy as np
import matplotlib.pyplot as plt
from price_pred import load, estimate_coef


c_reset = "\033[0m"
c_shape = "\033[92m"  # green
c_red = "\033[91m"
plt.style.use(['ggplot'])


def normalize_data(x):
    """Normalize the data"""
    return (x - np.mean(x)) / np.std(x)


# Define the cost/loss function
def mean_squared_error(actual, prediction):
    """Mean Squared Error"""
    return np.mean((actual - prediction) ** 2)


def gradient_descent(x, y, L = 1e-2, iter = 1000, stop_threshold = 1e-6):
    """Gradient Descent Algorithm"""
    w, b = 0, 0

    # Normalize the data
    x_norm = normalize_data(x)
    y_norm = normalize_data(y)

    costs = []
    weights = []

    for i in range(iter):
        y_pred = w * x_norm + b
        current_cost = mean_squared_error(y_norm, y_pred)

        if i > 0 and abs(costs[-1] - current_cost) <= stop_threshold:
            print(f"Convergence reached at iteration {c_red}{i+1}{c_reset}")
            break

        costs.append(current_cost)
        weights.append(w)

        # calculate gradients
        dw = - 2 * np.mean(x_norm * (y_norm - y_pred))        # theta1
        db = - 2 * np.mean(y_norm - y_pred)                   # theta0

        # update parameters
        w -= L * dw
        b -= L * db

    # Denormalize the parameters
    w = w * np.std(y) / np.std(x)
    b = np.mean(y) - w * np.mean(x)

    return w, b, costs, weights


def R_coeff(actual, predict):
    """Calculate the R-squared coefficient"""
    y_mean = np.mean(actual)

    ss_res = np.sum((predict - y_mean) ** 2)    # SS Residual : explained sum of squares
    ss_tot = np.sum((actual - y_mean) ** 2)     # SS Total : total sum of squares

    r_squared = ss_res / ss_tot
    return r_squared


def main():
    """ Main function """
    data = load("./data/data.csv")

    x = data['km']
    y = data['price']

    w_rl, b_rl = estimate_coef(x, y)
    w, b, costs, weights = gradient_descent(x, y)
    print(f"Estimated Weight: {c_shape}{w:.5f}{c_reset} Estimated Bias: {c_shape}{b:.0f}{c_reset}")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, w * x + b, color='red', label='Regression line', linewidth=1)
    plt.plot(x, w_rl * x + b_rl, color='green', label='Linear Regression', linestyle='dashed', linewidth=1)
    plt.title("Regression Line")
    plt.xlabel("km")
    x_ticks = [50000, 100000, 150000, 200000, 250000]
    x_ticks_label = ['50 k', '100 k', '150 k', '200 k', '250 k']
    plt.xticks(ticks=x_ticks, labels=x_ticks_label)
    plt.ylabel("price")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    r_squared = R_coeff(y, w * x + b)
    print(f"Model precision : {c_red}{r_squared:.3f}{c_reset}")
    print(f"Last cost : {c_red}{costs[-1]:.3f}{c_reset}")

if __name__ == "__main__":
    main()
