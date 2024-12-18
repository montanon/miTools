import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta


class GeneralizedBetaDistribution:
    def __init__(self, alpha, beta, a, b):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b

    def pdf(self, x):
        """Compute the pdf of the generalized Beta distribution."""
        numerator = (x - self.a) ** (self.alpha - 1) * (self.b - x) ** (self.beta - 1)
        denominator = beta(self.alpha, self.beta) * (self.b - self.a) ** (self.alpha + self.beta - 1)
        return numerator / denominator

    def plot_pdf(self):
        """Plot the pdf of the generalized Beta distribution."""
        x = np.linspace(self.a, self.b, 1000)
        y = [self.pdf(xi) for xi in x]
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label=f'Beta({self.alpha}, {self.beta}) on [{self.a}, {self.b}]')
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title('Generalized Beta Distribution')
        plt.legend()
        plt.grid()
        plt.show()
