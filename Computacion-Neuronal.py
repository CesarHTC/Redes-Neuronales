
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Crear los datos de entrenamiento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# Crear el modelo de red neuronal
modelo = Sequential()
modelo.add(Dense(4, input_dim=2, activation='sigmoid'))
modelo.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X, Y, epochs=200, batch_size=4)

# Evaluar el modelo
scores = modelo.evaluate(X, Y)
print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))
