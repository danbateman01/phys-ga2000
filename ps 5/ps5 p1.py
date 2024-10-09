#Newman 5.15
import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return 1 + np.tanh(2 * x)

def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def analytic_derivative(x):
    return 2 * (1 / np.cosh(2 * x))**2


x_values = np.linspace(-2, 2, 100)
numerical_derivatives = [central_difference(f, x) for x in x_values]
analytic_derivatives = analytic_derivative(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, numerical_derivatives, 'o', label="Numerical Derivative")
plt.plot(x_values, analytic_derivatives, label="Analytic Derivative", color='green')
plt.title("Comparison of Numerical and Analytic Derivative of f(x) = 1 + tanh(2x)")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
#plt.savefig("numerical&analytical")
plt.show()

