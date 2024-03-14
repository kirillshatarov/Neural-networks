import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

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

        decimal_output = np.argmax(output_layer_output)

        return decimal_output


# архитектура НС
input_size = 7
hidden_size = 7
output_size = 10
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
Y_train = np.eye(10)

# array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   0
#        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   1
#        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   2
#        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   3
#        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   4
#        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   5
#        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   6
#        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   7
#        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   8
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])  9

# обучение НС
losses = neural_network.train(X_train, Y_train, learning_rate=0.7, epochs=5000)



# TKINTER
def click(button):
    button['bg'] = "black" if button['bg'] == "white" else "white"
def start():
    btns_is_black = [int(b['bg'] == "black") for b in btns]

    # Проверка, что введенное состояние семисегментного индикатора соответствует известным числам
    if not any((btns_is_black == x).all() for x in X_train):
        label.config(text="Числа не существует")
        return

    X_test = np.array([btns_is_black])
    pred = neural_network.predict(X_test)
    print(pred)
    if pred is not None:
        label.config(text=f"Прогнозное значение: {int(pred)}")
    else:
        label.config(text="Ошибка при предсказании")

root = tk.Tk()
root.configure(bg='#FFD700')
root.geometry("600x550")

btns = [
    tk.Button(root, bg="white", width=2, height=11, command=lambda : click(btns[0])), # 0
    tk.Button(root, bg="white", width=20, height=1, command=lambda : click(btns[1])), # 1
    tk.Button(root, bg="white", width=2, height=11, command=lambda : click(btns[2])), # 2
    tk.Button(root, bg="white", width=2, height=11, command=lambda : click(btns[3])), # 3
    tk.Button(root, bg="white", width=20, height=1, command=lambda : click(btns[4])), # 4
    tk.Button(root, bg="white", width=2, height=11, command=lambda : click(btns[5])), # 5
    tk.Button(root, bg="white", width=20, height=1, command=lambda : click(btns[6]))  # 6
]

btns[0].place(x=201, y=30) # 0
btns[1].place(x=225, y=25) # 1
btns[2].place(x=375, y=30) # 2
btns[3].place(x=375, y=210) # 3
btns[4].place(x=225, y=368) # 4
btns[5].place(x=201, y=210) # 5
btns[6].place(x=225, y=190) # 6

btn_pred = tk.Button(root, text="Какое это число?", command=start, width=20, height=2, bg="#00FF00")
btn_pred.place(x=100, y=450)
label = tk.Label(root, text="Прогноз", width=20, height=2, bg="#00FF00")
label.place(x=350, y=450)
root.mainloop()


# график MSE
plt.plot(losses)
plt.show()