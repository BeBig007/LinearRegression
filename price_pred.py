import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


c_reset = "\033[0m"
c_shape = "\033[92m"
c_red = "\033[91m"
plt.style.use(['ggplot'])


def load(path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file and print its content."""
    try:
        data = pd.read_csv(path)
        print(f"Loading dataset {path}")
        return data
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"Error: {e}")
    return None


def model_pred(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Make predictions based on the regression line """
    return w * x + b


def plot_regression_line(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    """Plot the regression line """
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Regression line')
    x_ticks = [50000, 100000, 150000, 200000, 250000]
    x_ticks_label = ['50 k', '100 k', '150 k', '200 k', '250 k']
    plt.xticks(ticks=x_ticks, labels=x_ticks_label)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_line.png")
    plt.show()


def ask_input(w: float, b: float) -> tuple:
    """Ask for mileage input and calculate price based on a formula """
    while True:
        try:
            mileage = float(input("Enter the mileage (in km): "))
            if mileage < 0:
                print("Mileage cannot be negative. Please try again.")
                continue
            predicted_price = w * mileage + b
            print(f"Estimated price for {c_shape}{mileage:.0f}{c_reset}km is {c_shape}{predicted_price:.2f}{c_reset}$")
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def main() -> None:
    """Main function to run the program """
    if len(sys.argv) > 2:
        print(f"Usage: python {sys.argv[0]} weights.txt")
        sys.exit(1)

    if len(sys.argv) == 1:
        w, b = 0.0, 0.0
    if len(sys.argv) == 2:
        try:
            with open(sys.argv[1], "r") as file:
                lines = file.readlines()
                w = float(lines[0].split(",")[1])
                b = float(lines[1].split(",")[1])
        except FileNotFoundError:
            print(f"{c_red}Error: File not found{c_reset}")
            sys.exit(1)
        except Exception as e:
            print(f"{c_red}Error: {e}{c_reset}")
            sys.exit(1)

    data = load("./data/data.csv")
    if data is None:
        sys.exit(1)

    x = data['km'].values
    y = data['price'].values

    # Make predictions
    y_pred = model_pred(x, w, b)

    # Plot the regression line
    plot_regression_line(x, y, y_pred)

    # Ask for mileage input and calculate price
    ask_input(w, b)


if __name__ == "__main__":
    main()
