import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

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

    def train(self, X_train, Y_train, learning_rate, epochs):
        loss = []
        for epoch in range(epochs):
            # Прямое распространение ошибки
            input_layer = X_train
            hidden1_layer_input = np.dot(input_layer, self.input_weights) + self.bias1
            hidden1_layer_output = sigmoid(hidden1_layer_input)

            hidden2_layer_input = np.dot(hidden1_layer_output, self.hidden_weights) + self.bias2
            hidden2_layer_output = sigmoid(hidden2_layer_input)

            output_layer_input = np.dot(hidden2_layer_output, self.output_weights) + self.bias3
            output_layer_output = sigmoid(output_layer_input)

            loss.append(MSE(Y_train, output_layer_output))

            # Обратное распространение ошибки
            error = Y_train - output_layer_output
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

    def predict(self, input_values):
        input_layer = input_values
        hidden1_layer_input = np.dot(input_layer, self.input_weights) + self.bias1
        hidden1_layer_output = sigmoid(hidden1_layer_input)

        hidden2_layer_input = np.dot(hidden1_layer_output, self.hidden_weights) + self.bias2
        hidden2_layer_output = sigmoid(hidden2_layer_input)

        output_layer_input = np.dot(hidden2_layer_output, self.output_weights) + self.bias3
        output_layer_output = sigmoid(output_layer_input)

        binary_output = np.round(output_layer_output).astype(int)
        decimal_output = np.sum(binary_output * 2**np.arange(len(binary_output)-1, -1, -1))

        return binary_output, decimal_output


# архитектура НС
input_size = 7
hidden_size = 7
output_size = 4
neural_network = NeuralNetwork(input_size, hidden_size, output_size)

# Обучающие данные для 7-сегментного индикатора

#  ___1___
# |       |
# 0       2
# |___6___|
# |       |
# 5       3
# |___4___|


X_train = np.array([
    np.array([1,1,1,1,1,1,0]), # 0
    np.array([0,0,1,1,0,0,0]), # 1
    np.array([0,1,1,0,1,1,1]), # 2
    np.array([0,1,1,1,1,0,1]), # 3
    np.array([1,0,1,1,0,0,1]), # 4
    np.array([1,1,0,1,1,0,1]), # 5
    np.array([1,1,0,1,1,1,1]), # 6
    np.array([0,1,1,1,0,0,0]), # 7
    np.array([1,1,1,1,1,1,1]), # 8
    np.array([1,1,1,1,1,0,1]), # 9
])
Y_train = np.array([
    np.array([0,0,0,0]), # 0
    np.array([0,0,0,1]), # 1
    np.array([0,0,1,0]), # 2
    np.array([0,0,1,1]), # 3
    np.array([0,1,0,0]), # 4
    np.array([0,1,0,1]), # 5
    np.array([0,1,1,0]), # 6
    np.array([0,1,1,1]), # 7
    np.array([1,0,0,0]), # 8
    np.array([1,0,0,1]), # 9
])

# обучение НС
losses = neural_network.train(X_train, Y_train, learning_rate=0.3, epochs=10000)

# график MSE
plt.plot(losses)
plt.show()


# предсказывание
# [1,1,1,1,1,0,1] # 9
# [1,1,1,1,1,1,0] # 0
# [1,1,0,1,1,1,1] # 6
# [1,0,1,1,0,0,1] # 4

test_digit = np.array([1,0,1,1,0,0,1])  # 4
binary_output, decimal_output = neural_network.predict(test_digit)
print(f"Прогнозируемый вывод для тестовой цифры: \n binary - {binary_output}, decimal - {decimal_output}")
