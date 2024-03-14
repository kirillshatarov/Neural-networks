import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def minmax_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

def minmax_denormalize(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов
        self.input_weights = np.random.uniform(size=(input_size, hidden_size))
        self.bias1 = np.zeros(hidden_size)
        self.hidden_weights = np.random.uniform(size=(hidden_size, hidden_size))
        self.bias2 = np.zeros(hidden_size)
        self.output_weights = np.random.uniform(size=(hidden_size, output_size))
        self.bias3 = np.zeros(output_size)

        # Добавленные атрибуты для нормализации и денормализации
        self.min_X = None
        self.max_X = None
        self.min_Y = None
        self.max_Y = None

    def train(self, X_train, Y_train, learning_rate, epochs):
        # Мин-макс нормализация данных
        X_train_normalized, self.min_X, self.max_X = minmax_normalize(X_train)
        Y_train_normalized, self.min_Y, self.max_Y = minmax_normalize(Y_train)

        loss = []
        for epoch in range(epochs):
            # Прямое распространение ошибки
            input_layer = X_train_normalized
            hidden1_layer_input = np.dot(input_layer, self.input_weights) + self.bias1
            hidden1_layer_output = sigmoid(hidden1_layer_input)

            hidden2_layer_input = np.dot(hidden1_layer_output, self.hidden_weights) + self.bias2
            hidden2_layer_output = sigmoid(hidden2_layer_input)

            output_layer_input = np.dot(hidden2_layer_output, self.output_weights) + self.bias3
            output_layer_output = sigmoid(output_layer_input)

            loss.append(MSE(Y_train_normalized, output_layer_output))

            # Обратное распространение ошибки
            error = Y_train_normalized - output_layer_output
            d_output = error * sigmoid_derivative(output_layer_output)
            error_hidden2_layer = d_output.dot(self.output_weights.T)
            d_hidden2_layer = error_hidden2_layer * sigmoid_derivative(hidden2_layer_output)
            error_hidden1_layer = d_hidden2_layer.dot(self.hidden_weights.T)
            d_hidden1_layer = error_hidden1_layer * sigmoid_derivative(hidden1_layer_output)

            # Коррекция весов
            self.output_weights += hidden2_layer_output.T.dot(d_output) * learning_rate
            self.bias3 += np.sum(d_output, axis=0) * learning_rate
            self.hidden_weights += hidden1_layer_output.T.dot(d_hidden2_layer) * learning_rate
            self.bias2 += np.sum(d_hidden2_layer, axis=0) * learning_rate
            self.input_weights += input_layer.T.dot(d_hidden1_layer) * learning_rate
            self.bias1 += np.sum(d_hidden1_layer, axis=0) * learning_rate

        return loss

    def predict(self, X):
        # Нормализация входных данных
        X_normalized = (X - self.min_X) / (self.max_X - self.min_X)

        # Прямое распространение ошибки
        hidden1_layer_input = np.dot(X_normalized, self.input_weights) + self.bias1
        hidden1_layer_output = sigmoid(hidden1_layer_input)

        hidden2_layer_input = np.dot(hidden1_layer_output, self.hidden_weights) + self.bias2
        hidden2_layer_output = sigmoid(hidden2_layer_input)

        output_layer_input = np.dot(hidden2_layer_output, self.output_weights) + self.bias3
        output_layer_output = sigmoid(output_layer_input)

        # Денормализация предсказанных данных
        prediction = minmax_denormalize(output_layer_output, self.min_Y, self.max_Y)

        return prediction

# Загрузка данных
df = pd.read_csv('./USD_RUB.csv')
df['Цена'] = df['Цена'].str.replace(',', '.').astype(float)
data = df['Цена'].tolist()

plt.plot(data)  # График исходных данных
plt.xlabel('t')
plt.ylabel('value')
plt.title('Исходные данные')
plt.show()


# Обучающие данные
data = np.array(data)
window_size = 4
X_train = []
Y_train = []

for i in range(len(data) - window_size):
    x_window = data[i:i + window_size]
    y_label = data[i + window_size]
    X_train.append(x_window)
    Y_train.append(y_label)

X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape(-1, 1)
print(pd.DataFrame(zip(X_train, Y_train), columns=["X_train", "Y_train"]))

# Архитектура НС
input_size = 4
hidden_size = 4
output_size = 1
neural_network = NeuralNetwork(input_size, hidden_size, output_size)

# Обучение НС
losses = neural_network.train(X_train, Y_train, learning_rate=0.001, epochs=10000)

plt.plot(losses)  # график MSE
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('График MSE')
plt.show()

# Предсказание нового значения
X_for_forecasting = data[-4:]  # последние 4 значения для прогнозирования нового следующего значения
prediction = neural_network.predict(X_for_forecasting)
print(f'Последние 4 значения исходного ряда: {X_for_forecasting} \nСпрогнозированное значение: {prediction}')

# Создание графика
plt.plot(data, label='Исходные данные', color='blue')  # Линия с цветом blue
plt.scatter(len(data), prediction, color='red', label='Прогнозное значение')
# Соединение исходных данных с прогнозом
plt.plot([len(data) - 1, len(data)], [data[-1], prediction[0]], color='red', linestyle='--')

plt.title('Прогноз на графике с исходными данными')
plt.xlabel('t')
plt.ylabel('value')
plt.legend()
plt.show()