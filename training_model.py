import numpy as np
import matplotlib.pyplot as plt
from price_pred import load


c_reset = "\033[0m"
c_shape = "\033[92m"
c_red = "\033[91m"
plt.style.use(['ggplot'])


def normalize_data(x: np.ndarray) -> np.ndarray:
    """Normalize the data"""
    return (x - np.mean(x)) / np.std(x)


def mean_squared_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((actual - prediction) ** 2)


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    L: float = 1e-2,
    iter: int = 1000,
    stop_threshold: float = 1e-6
) -> tuple:
    """Gradient Descent Algorithm"""
    w, b = 0.0, 0.0

    # Normalize the data
    x_norm = normalize_data(x)
    y_norm = normalize_data(y)

    costs = []

    for i in range(iter):
        y_pred = w * x_norm + b
        current_cost = mean_squared_error(y_norm, y_pred)

        if i > 0 and abs(costs[-1] - current_cost) <= stop_threshold:
            print(f"Convergence reached at iteration {c_red}{i+1}{c_reset}")
            break

        costs.append(current_cost)

        # calculate gradients
        dw = - 2 * np.mean(x_norm * (y_norm - y_pred))
        db = - 2 * np.mean(y_norm - y_pred)

        # update parameters
        w -= L * dw
        b -= L * db

    # Denormalize the parameters
    w = w * np.std(y) / np.std(x)
    b = np.mean(y) - w * np.mean(x)

    return w, b, costs


def R_coeff(actual: np.ndarray, predict: np.ndarray) -> float:
    """Calculate the R-squared coefficient"""
    y_mean = np.mean(actual)

    ss_res = np.sum((predict - y_mean) ** 2)
    ss_tot = np.sum((actual - y_mean) ** 2)

    return ss_res / ss_tot


def main() -> None:
    """ Main function """
    data = load("./data/data.csv")
    if data is None:
        return

    x = data['km'].values
    y = data['price'].values

    # Estimate the coefficients
    w, b, costs = gradient_descent(x, y)

    print(f"Training completed: w = {c_shape}{w:.5f}{c_reset}, b = {c_shape}{b:.0f}{c_reset}")

    #save results in weights.txt
    with open("weights.txt", "w") as file:
        file.write(f"w,{w}\n")
        file.write(f"b,{b}\n")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, w * x + b, color='red', label='Regression line')
    plt.title("Regression Line")
    plt.xlabel("km")
    x_ticks = [50000, 100000, 150000, 200000, 250000]
    x_ticks_label = ['50 k', '100 k', '150 k', '200 k', '250 k']
    plt.xticks(ticks=x_ticks, labels=x_ticks_label)
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_line.png")
    plt.show()

    # Calculate the R-squared coefficient
    r2 = R_coeff(y, w * x + b)
    print(f"Model precision : {c_red}{r2:.3f}{c_reset}")


if __name__ == "__main__":
    main()
