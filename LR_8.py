import numpy as np
import matplotlib.pyplot as plt

class KohonenMap:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self.init_weights()

    def init_weights(self):  # инициализация весов
        initial_weights = np.random.rand(self.input_size, self.output_size)
        return initial_weights

    def update_weights(self, input_data, winner_index, learning_rate):  # обновление весов
        d_weights = learning_rate * (input_data - self.weights[:, winner_index])
        self.weights[:, winner_index] += d_weights

    def find_winner_neuron(self, input_data):  # поиск нейрона победителя
        distances = np.linalg.norm(self.weights - input_data[:, np.newaxis], axis=0)  # рассчёт Евклидова расстояния между весами и входным образом.
        winner_index = np.argmin(distances)  # индекс минимального значения (расстояния)
        return winner_index

    def train(self, data, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(data.shape[0]):  # цикл по всем образцам в наборе данных
                input_data = data[i, :]
                winner_index = self.find_winner_neuron(input_data)  # "победивший" нейрон для текущего входного образца
                self.update_weights(input_data, winner_index, learning_rate)  # обновление весов карты Кохонена на основе текущего входного образа и "победившего" нейрона

    def visualize_clusters(self, data):
        s = np.linalg.norm(self.weights.T - data[:, np.newaxis], axis=2)
        clusters = np.argmin(s, axis=1)
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'pink', 'orange', 'brown', 'purple', 'silver']
        for i in range(data.shape[0]):
            plt.scatter(data[i, 0], data[i, 1], color=colors[clusters[i]])
        plt.scatter(self.weights[0, :], self.weights[1, :], marker='*', s=200, color='black', label='Центр кластера')
        plt.legend()
        plt.show()

# Данные
np.random.seed(42)
# Генерируем данные, используя нормальное распределение (рост и вес)
data = np.random.normal(loc=[170, 70], scale=[30, 15], size=(800, 2))

normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))  # нормализация к шкале [0, 1]

num_clusters = 3  # число кластеров
epochs = 3000
learning_rate = 0.01

kohonen_map = KohonenMap(input_size=2, output_size=num_clusters)
kohonen_map.visualize_clusters(normalized_data)  # график до обучения НС

kohonen_map.train(normalized_data, epochs, learning_rate)
kohonen_map.visualize_clusters(normalized_data)  # график после обучения НС

