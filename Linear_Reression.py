import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:

    def __init__(self, learning_rate, iter_max):
        self.prices = []
        self.years = []
        self.learning_rate = learning_rate
        self.iter_max = iter_max
        self.datalen = 0
        self.load_data()
        self.y = np.zeros(self.datalen-1)
        self.m = 0
        self.b = 0
        self.m_history = []
        self.b_history = []
        self.j_history = []

    def load_data(self):
        with open('D:\ML\MyProgress\Linear Regression\First-Linear-Regression\Salary Dataset (Linear Regression)\\Salary_dataset.csv', 'r') as file:
            data = file.readlines()
            for line in data:
                self.datalen += 1
                if self.datalen == 1:
                    continue
                else:
                    line = line.strip().split(',')
                    year = float(line[0])
                    price = float(line[1])
                    year = round(year, 3)
                    price = round(price, 3)
                    self.years.append(year)
                    self.prices.append(price)

    def compute_cost(self, m, b):
        predictions = m * np.array(self.years) + b
        return 1 / (2 * self.datalen) * np.sum((predictions - self.prices) ** 2)

    def fit(self):
        self.y = self.m * np.array(self.years) + self.b  # Calcular las predicciones

        it = 0
        while it < self.iter_max:
            error = self.compute_cost(self.m, self.b)
            if error < 0.0001:
                break
            # Actualizar m y b utilizando el gradiente descendente
            self.j_history.append(error)
            self.m_history.append(self.m)
            self.b_history.append(self.b)
            self.m = self.m - self.learning_rate * 1 / self.datalen * np.sum((self.y - self.prices) * np.array(self.years))
            self.b = self.b - self.learning_rate * 1 / self.datalen * np.sum(self.y - self.prices)
            
            # Recalcular y con los nuevos m y b
            self.y = self.m * np.array(self.years) + self.b
            
            it += 1

        return self.y, self.j_history, self.m_history, self.b_history

    def get_prediction (self, years):
        y = self.m * years + self.b
        return y

def main():
    lr = LinearRegression(0.001, 1000)
    plt.scatter(lr.years, lr.prices)
    plt.title('Housing Prices')
    plt.xlabel('Year')
    plt.ylabel('Price')

    # Calcular los valores ajustados
    fitted_values, j, m , b = lr.fit()

    # Graficar los datos y la línea de regresión lineal
    plt.plot(lr.years, fitted_values, color='red')

    plt.legend(['Data', 'Linear Regression'])
    plt.show()

    print(lr.get_prediction(3))


    plt.plot(m, j, color='red')

    plt.legend(['m', 'j'])
    plt.show()


    # Crear una cuadrícula de valores para m y b
    m_vals = np.linspace(min(m)-1000, max(m)+1000, 300)
    b_vals = np.linspace(min(b)-1000, max(b)+1000, 300)
    M, B = np.meshgrid(m_vals, b_vals)
    J = np.zeros_like(M)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            J[i, j] = lr.compute_cost(M[i, j], B[i, j])

    # Graficar la superficie del costo
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, B, J, cmap='viridis')
    ax.set_xlabel('m')
    ax.set_ylabel('b')
    ax.set_zlabel('J')
    plt.title('Cost function J(m, b)')
    plt.show()

if __name__ == '__main__':
    main()
