import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


c_reset = "\033[0m"
c_shape = "\033[92m"  # green
c_red = "\033[91m"


def load(path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file and print its content."""

    try:
        assert isinstance(path, str), "the input must be string"
        data = pd.read_csv(path)
        print(f"Loading dataset {path}")

        # plt.figure()
        # plt.scatter(data['km'], data['price'])
        # plt.xlabel('km')
        # x_ticks = [50000, 100000, 150000, 200000, 250000]
        # x_ticks_label = ['50 k', '100 k', '150 k', '200 k', '250 k']
        # plt.xticks(ticks=x_ticks, labels=x_ticks_label)
        # plt.ylabel('price')
        # plt.show()

        return data

    except AssertionError as msg:
        print(f"AssertionError: {msg}")

    except Exception as e:
        print(f"Error: {e}")

    return None


def estimate_coef(x, y):
    """ """
    # Calculating Number of Observations
    n = np.size(x)

    # Calculating Means
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculating Cross-Deviation and Deviation about x
    ss_xy = np.sum(y * x) - n * mean_x * mean_y
    ss_xx = np.sum(x * x) - n * mean_x * mean_x

    # Calculating Regression Coefficients
    b_1 = ss_xy / ss_xx             # y = b[1] * x + b[0]
    b_0 = mean_y - b_1 * mean_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    """ """
    y_pred = b[0] + b[1] * x

    # plt.figure()
    # plt.scatter(x, y)
    # plt.plot(x, y_pred, color="red")
    # plt.xlabel("km")
    # x_ticks = [50000, 100000, 150000, 200000, 250000]
    # x_ticks_label = ['50 k', '100 k', '150 k', '200 k', '250 k']
    # plt.xticks(ticks=x_ticks, labels=x_ticks_label)
    # plt.ylabel("price")
    # plt.show()


def ask_input(b):
    """Ask for mileage input and calculate price based on a formula """
    max_km = -b[0] / b[1]

    while True:
        try:
            mileage = float(input(f"Enter your mileage (between 0 and {round(max_km)}): "))
            if 0 < mileage < max_km:
                break
            elif mileage == 0 or mileage == max_km:
                print("Mileage cannot be exactly 0 or the maximum value.")
            else:
                print(f"{c_red}Please enter a number between 0 and {round(max_km)}.{c_reset}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    new_price = (b[0] + b[1] * mileage)

    print(f"For a car with {c_shape}{mileage:.0f}{c_reset} km, price will be {c_shape}{new_price:.0f}{c_reset}$")

    return mileage, new_price


def main():
    data = load("./data/data.csv")

    x = data['km']
    y = data['price']

    b = [0, 0]
    b = estimate_coef(x, y)
    print("y(x) = a * x + b")
    print("y(x) = {:.5f}".format(b[1]), "* x + {:.0f}".format(b[0]))
    plot_regression_line(x, y, b)

    ask_input(b)


if __name__ == "__main__":
    main()
