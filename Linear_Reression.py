import numpy as np
import matplotlib.pyplot as plt

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

    def load_data(self):
        with open('D:\\ML\\MyProgress\\Linear Regression\\Salary Dataset (Linear Regression)\\Salary_dataset.csv', 'r') as file:
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

    def fit(self):
        self.y = self.m * np.array(self.years) + self.b  # Calcular las predicciones

        it = 0
        while it < self.iter_max:
            error = 1 / (2 * self.datalen) * np.sum((self.y - self.prices) ** 2)
            if error < 0.0001:
                break
            # Actualizar m y b utilizando el gradiente descendente
            self.m = self.m - self.learning_rate * 1 / self.datalen * np.sum((self.y - self.prices) * np.array(self.years))
            self.b = self.b - self.learning_rate * 1 / self.datalen * np.sum(self.y - self.prices)
            
            # Recalcular y con los nuevos m y b
            self.y = self.m * np.array(self.years) + self.b
            
            it += 1

        return self.y


    def get_prediction (self,years):

        y = self.m * years + self.b
        return y

def main():
    lr = LinearRegression(0.001, 1000)
    plt.scatter(lr.years, lr.prices)
    plt.title('Housing Prices')
    plt.xlabel('Year')
    plt.ylabel('Price')

    # Calcular los valores ajustados
    fitted_values = lr.fit()

    # Graficar los datos y la línea de regresión lineal
    plt.plot(lr.years, fitted_values, color='red')

    plt.legend(['Data', 'Linear Regression'])
    plt.show()


    print(lr.get_prediction(3))

if __name__ == '__main__':
    main()
