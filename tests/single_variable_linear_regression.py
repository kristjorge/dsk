import numpy as np
from matplotlib import pyplot as plt
import dsk.linear_model as linmod


def main():
    X = np.array([np.random.random()*5 for _ in range(20)])
    y = np.array([(x + np.random.random()*5) for x in X])

    linear_regression = linmod.LinearRegression()
    linear_regression.fit(X, y)

    a = linear_regression.coefficients[0]
    b = linear_regression.coefficients['intercept']
    f = np.vectorize(lambda x: a*x + b)
    plt.scatter(X, y, c='b')
    plt.plot(X, f(X), c='r')
    plt.show()


if __name__ == '__main__':
    main()