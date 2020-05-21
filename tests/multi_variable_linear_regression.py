import numpy as np
from matplotlib import pyplot as plt
import dsk.linear_model as linmod


def main():
    X = np.array([[np.random.random()] for _ in range(50)])
    y = np.array([x + np.random.random() for x in X[:, 0]])

    linear_regression = linmod.LinearRegression(learning_rate=0.1, epochs=500)
    linear_regression.fit(X, y)

    a = linear_regression.coefficients[0].value
    b = linear_regression.coefficients[1].value
    f = np.vectorize(lambda x: a*x + b)
    plt.scatter(X[:, 0], y, c='r')
    plt.plot(np.arange(-5, 5, 0.1), f(np.arange(-5,5,0.1)))
    plt.show()

    prediction = linear_regression.predict(np.array([-3]))
    print(prediction)


if __name__ == '__main__':
    main()
