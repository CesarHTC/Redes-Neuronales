import numpy as np
import matplotlib.pyplot as plt

# Definir las funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Generar los datos de entrada
x = np.linspace(-10, 10, 100)

# Crear una figura con 3 subgráficos
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Graficar la función de activación sigmoidal
y_sigmoid = sigmoid(x)
axs[0].plot(x, y_sigmoid)
axs[0].set_title('Sigmoid')

# Graficar la función de activación ReLU
y_relu = relu(x)
axs[1].plot(x, y_relu)
axs[1].set_title('ReLU')

# Graficar la función de activación tanh
y_tanh = tanh(x)
axs[2].plot(x, y_tanh)
axs[2].set_title('Tanh')

# Crear una red neuronal simple
def neural_network(input_data):
    hidden_layer = np.dot(input_data, np.random.rand(1, 10))
    output = np.dot(hidden_layer, np.random.rand(10, 1))
    return output

# Ejecutar la red neuronal y graficar la salida utilizando la función de activación sigmoidal
y_output_sigmoid = sigmoid(neural_network())
axs[0].plot(x, y_output_sigmoid)

# Ejecutar la red neuronal y graficar la salida utilizando la función de activación ReLU
y_output_relu = relu(neural_network(x))
axs[1].plot(x, y_output_relu)

# Ejecutar la red neuronal y graficar la salida utilizando la función de activación tanh
y_output_tanh = tanh(neural_network(x))
axs[2].plot(x, y_output_tanh)

# Ajustar los parámetros de las figuras
plt.tight_layout()

# Mostrar la figura
plt.show()
