import numpy as np
from matplotlib import pyplot as plt
import dsk.linear_model as linmod


def main():
    X = np.array([np.random.random()*5 for _ in range(50)])
    y = np.array([(x + np.random.random()*0.95) for x in X])

    linear_regression = linmod.LinearRegression(learning_rate=0.1, epochs=500)
    linear_regression.fit(X, y)

    a = linear_regression.coefficients[0].value
    b = linear_regression.coefficients[1].value
    f = np.vectorize(lambda x: a*x + b)
    plt.scatter(X, y, c='r')
    plt.plot(X, f(X))
    # plt.show()

    prediction = linear_regression.predict(np.array([25]))
    print(prediction)


if __name__ == '__main__':
    main()
