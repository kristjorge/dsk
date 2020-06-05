import numpy as np
from matplotlib import pyplot as plt
import dsk.linear_model as linmod
from dsk.data_sets import two_variable_linear_model


def main():
    X = two_variable_linear_model.iloc[:, :2].values
    y = two_variable_linear_model.iloc[:, 2].values

    linear_regression = linmod.LinearRegression(learning_rate=0.05, epochs=200)
    linear_regression.fit(X, y, mini_batch_size=None)

    a = linear_regression.coefficients[0].value
    b = linear_regression.coefficients[1].value
    c = linear_regression.coefficients[2].value
    f = np.vectorize(lambda x, y: a*x + b*y + c)

    print('a: {}'.format(a))
    print('b: {}'.format(b))
    print('c: {}'.format(c))
    print('R squared: {}'.format(linear_regression.R))

    plt.figure()
    plt.subplot(221)
    plt.plot(linear_regression.coefficients[0].log)
    plt.title('A')
    plt.subplot(222)
    plt.plot(linear_regression.coefficients[1].log)
    plt.title('B')
    plt.subplot(223)
    plt.plot(linear_regression.coefficients[2].log)
    plt.title('Intercept')
    plt.subplot(224)
    plt.plot(linear_regression.loss)
    plt.title('Loss')
    plt.show()


if __name__ == '__main__':
    main()
